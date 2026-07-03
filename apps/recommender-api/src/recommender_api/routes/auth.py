"""Authentication routes for recommender-api."""

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from common.db.models.users import AppUser
from recommender_api.runtime import InferenceRuntime
from recommender_api.dependencies import get_current_user
from recommender_api.settings import RecommenderApiSettings
from common.db.repositories.ratings import count_user_ratings
from common.db.repositories.users import create_user, get_user_by_username
from fastapi import APIRouter, HTTPException, Request, Response, status

from recommender_api.schemas import (
    AuthUserResponse,
    LoginRequest,
    RegisterRequest,
    UserStatusResponse,
)
from recommender_api.services.auth import (
    create_user_session,
    hash_password,
    logout_session,
    verify_password,
)


def _build_auth_response(session: Session, user: AppUser, settings: RecommenderApiSettings) -> AuthUserResponse:
    """
    Build the authenticated user summary with rating counts. 
    Tells the frontend how many movies the user has rated and 
    whether they are eligible to receive recommendations.
    """
    # Get the number of ratings the user has made.
    rating_count = count_user_ratings(session, user.user_id)
    # Build the authenticated user response.
    return AuthUserResponse(
        user_id=user.user_id,
        username=user.username,
        rating_count=rating_count,
        can_recommend=rating_count >= settings.min_ratings_for_recommend,
    )


def _set_session_cookie(response: Response, session_id: str, settings: RecommenderApiSettings) -> None:
    """ Attach the httpOnly session cookie to a response. """
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_id,
        httponly=True,
        max_age=settings.session_ttl_seconds,
        samesite="lax",
    )


def create_auth_router(runtime: InferenceRuntime, settings: RecommenderApiSettings) -> APIRouter:
    """
    Create authentication routes and wire them to the runtime dependencies.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.
    settings: Recommender API settings.

    ============================ Returns ============================
    Configured FastAPI router.
    """
    # Create the router.
    router = APIRouter(prefix="/v1/auth", tags=["auth"])

    # Register a new app user and start a login session.
    @router.post("/register", response_model=AuthUserResponse)
    def register(request_body: RegisterRequest, response: Response) -> AuthUserResponse:
        """
        Register a new app user and start a login session.

        Do this by:
        1. Rejecting duplicate usernames.
        2. Hashing the password and inserting the app_users row.
        3. Creating a Postgres session and setting the session cookie.
        """
        session = runtime.session_factory()
        try:
            # Check if the username already exists.
            if get_user_by_username(session, request_body.username) is not None:
                runtime.metrics.record_error("register", "duplicate_username")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already exists",
                )

            # Create the user.
            user = create_user(
                session,
                username=request_body.username,
                password_hash=hash_password(request_body.password),
            )
            # Create a user session.
            session_id = create_user_session(session, user, settings)
            # Commit the session.
            session.commit()
            # Set the session cookie.
            _set_session_cookie(response, session_id, settings)
            # Record the request.
            runtime.metrics.record_request("register", "success")
            # Return the authenticated user response.
            return _build_auth_response(session, user, settings)

        except IntegrityError as exc:
            session.rollback()
            runtime.metrics.record_error("register", "integrity_error")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            ) from exc
        
        except HTTPException:
            session.rollback()
            raise
        
        except Exception:
            session.rollback()
            runtime.metrics.record_error("register", "internal_error")
            raise
        
        finally:
            session.close()

    @router.post("/login", response_model=AuthUserResponse)
    def login(request_body: LoginRequest, response: Response) -> AuthUserResponse:
        """
        Verify username/password and start a new login session.

        Do this by:
        1. Loading the user by normalized username.
        2. Verifying the bcrypt password hash.
        3. Creating a Postgres session and setting the session cookie.
        """
        session = runtime.session_factory()

        # Try to load the user by normalized username.
        try:
            # Get the user by normalized username.
            user = get_user_by_username(session, request_body.username)
            # If the user is not found or the password is invalid, raise an error.
            if user is None or not verify_password(request_body.password, user.password_hash):
                runtime.metrics.record_error("login", "invalid_credentials")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                )

            # Create a user session.
            session_id = create_user_session(session, user, settings)
            # Commit the session.
            session.commit()
            # Set the session cookie.
            _set_session_cookie(response, session_id, settings)
            # Record the request.
            runtime.metrics.record_request("login", "success")
            # Return the authenticated user response.
            return _build_auth_response(session, user, settings)

        except HTTPException:
            session.rollback()
            raise

        except Exception:
            session.rollback()
            runtime.metrics.record_error("login", "internal_error")
            raise
        
        finally:
            session.close()

    @router.post("/logout")
    def logout(request: Request, response: Response) -> dict[str, str]:
        """
        Delete the current session and clear the session cookie.

        Do this by:
        1. Deleting the session from the database.
        2. Clearing the session cookie.
        3. Recording the request.
        """
        session_id = request.cookies.get(settings.session_cookie_name)
        session = runtime.session_factory()

        # Try to delete the session and clear the session cookie.
        try:
            # If the session id is found, delete the session and clear the session cookie.
            if session_id:
                logout_session(session, session_id)
                session.commit()
         
            # Clear the session cookie.
            response.delete_cookie(settings.session_cookie_name)
         
            # Record the request.
            runtime.metrics.record_request("logout", "success") 
            return {"status": "logged_out"}
        
        except Exception:
            session.rollback()
            runtime.metrics.record_error("logout", "internal_error")
            raise
        
        finally:
            session.close()

    @router.get("/status", response_model=UserStatusResponse)
    def user_status(username: str) -> UserStatusResponse:
        """
        Check whether a username exists and how many movies that user has rated.

        Do this by:
        1. Loading the user by normalized username.
        2. Checking if the user exists.
        3. Getting the number of ratings the user has made.
        4. Recording the request.
        5. Returning the user status response.
        """
        session = runtime.session_factory()
        try:
            # Get the user by normalized username.
            user = get_user_by_username(session, username)

            # If the user is not found, record the request and return the user status response.
            if user is None:
                runtime.metrics.record_request("status", "success")
                return UserStatusResponse(exists=False)

            # Get the number of ratings the user has made.
            rating_count = count_user_ratings(session, user.user_id)
            # Record the request.
            runtime.metrics.record_request("status", "success")

            # Return the user status response.
            return UserStatusResponse(
                exists=True,
                user_id=user.user_id,
                rating_count=rating_count,
                can_recommend=rating_count >= settings.min_ratings_for_recommend,
            )

        except Exception:
            runtime.metrics.record_error("status", "internal_error")
            raise
        
        finally:
            session.close()

    @router.get("/me", response_model=AuthUserResponse)
    def me(request: Request) -> AuthUserResponse:
        """
        Return the authenticated user's profile and rating readiness.

        Do this by:
        1. Getting the current user from the request.
        2. Creating a new session.
        3. Recording the request.
        4. Returning the authenticated user response.
        """
        # Get the current user from the request.    
        user = get_current_user(request, runtime, settings)
        session = runtime.session_factory()

        # Try to record the request and return the authenticated user response.
        try:
            runtime.metrics.record_request("me", "success")
            return _build_auth_response(session, user, settings)
        finally:
            session.close()

    return router
