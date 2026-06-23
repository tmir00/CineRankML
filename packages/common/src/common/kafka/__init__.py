"""Shared Kafka client helpers for producers and consumers."""

from common.kafka.producer import KafkaEventProducer
from common.kafka.serde import decode_json, encode_event
from common.kafka.topics import ALL_TOPICS, RATINGS_TOPIC, TAGS_TOPIC
from common.kafka.csv_checkpoint import read_csv_checkpoint, save_csv_checkpoint
from common.kafka.consumer import GracefulShutdown, KafkaEventConsumer, run_consumer_loop

__all__ = [
    "ALL_TOPICS",
    "RATINGS_TOPIC",
    "TAGS_TOPIC",
    "GracefulShutdown",
    "KafkaEventConsumer",
    "KafkaEventProducer",
    "decode_json",
    "encode_event",
    "read_csv_checkpoint",
    "run_consumer_loop",
    "save_csv_checkpoint",
]
