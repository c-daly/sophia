"""Smoke tests for OpenTelemetry instrumentation in Sophia."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from logos_observability import setup_telemetry, get_tracer


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the global tracer provider between tests."""
    yield
    trace.set_tracer_provider(TracerProvider())


def test_sophia_telemetry_setup():
    """Verify setup_telemetry configures a working TracerProvider."""
    provider = setup_telemetry(service_name="sophia", export_to_console=False)
    assert provider is not None

    tracer = get_tracer("sophia.test")
    with tracer.start_as_current_span("sophia.test_span") as span:
        span.set_attribute("test.key", "test_value")


def test_sophia_spans_have_correct_service_name():
    """Verify spans carry the correct service.name resource attribute."""
    setup_telemetry(service_name="sophia", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = get_tracer("sophia.planner")
    with tracer.start_as_current_span("sophia.plan") as span:
        span.set_attribute("plan.goal", "test")

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    assert spans[0].name == "sophia.plan"
    assert spans[0].resource.attributes["service.name"] == "sophia"


def test_sophia_nested_spans():
    """Verify nested spans maintain parent-child relationships."""
    setup_telemetry(service_name="sophia", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = get_tracer("sophia.api")
    with tracer.start_as_current_span("sophia.plan") as parent:  # noqa: F841
        with tracer.start_as_current_span("sophia.simulate") as child:
            child.set_attribute("simulation.steps", 5)

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child_span = next(s for s in spans if s.name == "sophia.simulate")
    parent_span = next(s for s in spans if s.name == "sophia.plan")
    assert child_span.parent.span_id == parent_span.context.span_id
