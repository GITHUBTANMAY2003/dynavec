"""Edge weights and properties for the DynamoDB-backed graph."""

from datetime import datetime, timedelta
from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

from dynavec import Dynavec
from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore


@pytest.fixture
def store():
    with mock_aws():
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )
        session.client("dynamodb").create_table(
            TableName="test-table",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        config = DynavecConfig(
            vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
        )
        yield GraphStore(config, boto_session=session)


def test_add_edge_stores_weight_and_props_without_mutating_the_caller(store):
    timestamp = "2026-09-29T12:34:56Z"
    props = {"kind": "citation", "created_at": timestamp}

    store.add_edge("ns", "source", "cites", "target", weight=2.75, props=props)
    props["kind"] = "changed-after-write"

    edge = store.get_node("ns", "source")["edges"][0]
    assert edge["weight"] == Decimal("2.75")
    assert edge["props"] == {"kind": "citation", "created_at": timestamp}
    assert isinstance(edge["props"]["created_at"], str)
    parsed_timestamp = datetime.fromisoformat(edge["props"]["created_at"].replace("Z", "+00:00"))
    assert parsed_timestamp.utcoffset() == timedelta(0)


def test_add_edge_defaults_weight_and_props(store):
    store.add_edge("ns", "source", "relates_to", "target")

    edge = store.get_node("ns", "source")["edges"][0]
    assert edge["weight"] == Decimal("1.0")
    assert edge["props"] == {}


def test_neighbors_preserves_unweighted_order_and_filters_weighted_results(store):
    store.add_edge("ns", "source", "related", "low", weight=1.0)
    store.add_edge("ns", "source", "related", "high", weight=4.0)
    store.add_edge("ns", "source", "related", "threshold", weight=2.0)
    store.add_edge("ns", "source", "other", "other-relation", weight=10.0)

    assert store.neighbors("ns", "source") == [
        "low",
        "high",
        "threshold",
        "other-relation",
    ]
    assert store.neighbors("ns", "source", with_weights=True) == [
        ("other-relation", 10.0),
        ("high", 4.0),
        ("threshold", 2.0),
        ("low", 1.0),
    ]
    assert store.neighbors("ns", "source", "related", min_weight=2.0) == [
        "high",
        "threshold",
    ]
    assert store.neighbors("ns", "source", "related", min_weight=2.0, with_weights=True) == [
        ("high", 4.0),
        ("threshold", 2.0),
    ]


def test_neighbors_treats_legacy_edges_without_weight_as_one(store):
    store.add_node("ns", "source")
    store.add_node("ns", "legacy")
    store._table.update_item(
        Key={"pk": store._node_pk("ns", "source")},
        UpdateExpression="SET edges = :edges",
        ExpressionAttributeValues={
            ":edges": [{"relation": "legacy", "target": "legacy"}],
        },
    )

    assert store.neighbors("ns", "source", with_weights=True) == [("legacy", 1.0)]
    assert store.neighbors("ns", "source", min_weight=1.0) == ["legacy"]
    assert store.neighbors("ns", "source", min_weight=1.01) == []


class RecordingGraph:
    def __init__(self):
        self.calls = []

    def add_edge(self, ns, src, relation, dst, *, weight=1.0, props=None):
        self.calls.append((ns, src, relation, dst, weight, props))


def test_graph_add_edge_forwards_weight_props_and_bidirectional_values():
    graph = RecordingGraph()
    client = Dynavec.__new__(Dynavec)
    client._graph_store = graph
    props = {"created_at": "2026-09-29T12:34:56Z"}

    client.graph_add_edge(
        "source",
        "connects",
        "target",
        namespace="tenant",
        bidirectional=True,
        weight=3.5,
        props=props,
    )

    assert graph.calls == [
        ("tenant", "source", "connects", "target", 3.5, props),
        ("tenant", "target", "connects", "source", 3.5, props),
    ]
