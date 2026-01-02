"""Feedback emission system for Sophia → Hermes communication."""

from sophia.feedback.config import FeedbackConfig
from sophia.feedback.dispatcher import FeedbackDispatcher
from sophia.feedback.models import FeedbackPayload, StateDiff, StepResult
from sophia.feedback.queue import FeedbackQueue
from sophia.feedback.worker import FeedbackWorker

__all__ = [
    "FeedbackConfig",
    "FeedbackDispatcher",
    "FeedbackPayload",
    "FeedbackQueue",
    "FeedbackWorker",
    "StateDiff",
    "StepResult",
]
