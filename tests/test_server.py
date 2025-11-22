import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import research_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_server import app, state
from research_defs import SearchResultEntry, MetadataEntry, TextLibraryEntry, DocumentRepresentationEntry

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_stores():
    # Mock the GlobalState
    state.ds = MagicMock()
    state.vs = MagicMock()
    
    # Setup some default return values
    state.ds.text_library = {"hash1": "doc1", "hash2": "doc2"}
    state.vs.config = {"embeddings_model_name": "test_model"}
    state.vs.model_list = [{"model_name": "test_model", "enabled": True}]
    
    yield
    
    state.ds = None
    state.vs = None

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["document_count"] == 2
    assert data["active_model"] == "test_model"

def test_search():
    # Mock search results
    mock_entry = TextLibraryEntry(source_name="test", descriptor="test_doc.md", text="This is a match")
    mock_result = SearchResultEntry(
        cosine=0.85,
        hash="hash1",
        chunk_index=0,
        entry=mock_entry,
        text="This is a match",
        significance=[1.0]
    )
    state.vs.search.return_value = [mock_result]
    
    # Mock metadata
    mock_meta = MetadataEntry(
        uuid="123",
        representations=[],
        authors=[],
        identifiers=[],
        languages=[],
        context="",
        creation_date="2023-01-01",
        publication_date="",
        publisher="",
        series="",
        tags=[],
        title="Test Doc",
        title_sort="",
        normalized_filename="",
        description="",
        icon=""
    )
    state.ds.get_metadata.return_value = mock_meta

    response = client.get("/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["descriptor"] == "test_doc.md"
    assert data[0]["score"] == 0.85
    assert data[0]["metadata"]["title"] == "Test Doc"

def test_ksearch():
    # Mock ksearch results
    mock_entry = TextLibraryEntry(source_name="test", descriptor="test_doc.md", text="Keyword match")
    mock_result = SearchResultEntry(
        cosine=1.0,
        hash="hash1",
        chunk_index=0,
        entry=mock_entry,
        text="Keyword match"
    )
    state.ds.keyword_search.return_value = [mock_result]
    
    mock_meta = MetadataEntry(
        uuid="123",
        representations=[],
        authors=[],
        identifiers=[],
        languages=[],
        context="",
        creation_date="2023-01-01",
        publication_date="",
        publisher="",
        series="",
        tags=[],
        title="Test Doc",
        title_sort="",
        normalized_filename="",
        description="",
        icon=""
    )
    state.ds.get_metadata.return_value = mock_meta

    response = client.get("/ksearch?q=keyword")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["descriptor"] == "test_doc.md"

def test_timeline():
    # Mock timeline results
    mock_event = {
        'jd_event': 2451545.0, # 2000-01-01
        'eventdata': 'Event Data'
    }
    state.ds.tl.search_events.return_value = [mock_event]
    state.ds.tl.get_date_string_from_event.return_value = "2000-01-01"
    state.ds.tl.get_event_text.return_value = "Test Event"

    response = client.get("/timeline?time=2000")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["date"] == "2000-01-01"
    assert data[0]["event"] == "Test Event"

def test_document_found():
    mock_meta = MetadataEntry(
        uuid="123",
        representations=[],
        authors=[],
        identifiers=[],
        languages=[],
        context="",
        creation_date="2023-01-01",
        publication_date="",
        publisher="",
        series="",
        tags=[],
        title="Test Doc",
        title_sort="",
        normalized_filename="",
        description="",
        icon=""
    )
    state.ds.get_metadata.return_value = mock_meta
    state.ds.get_path_from_descriptor.return_value = "/tmp/test_doc.md"
    
    # Mock file open? Or just check response structure
    # Since we can't easily mock open() globally without side effects, 
    # we'll just check that it tries to read.
    # But the endpoint checks os.path.exists.
    
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "File Content"
            
            response = client.get("/document/test_doc.md")
            assert response.status_code == 200
            data = response.json()
            assert data["descriptor"] == "test_doc.md"
            assert data["content"] == "File Content"

def test_document_not_found():
    state.ds.get_metadata.return_value = None
    response = client.get("/document/missing.md")
    assert response.status_code == 404

def test_sync():
    state.ds.import_local.return_value = True
    # We need to mock reload too, otherwise it tries to create real objects
    with patch.object(state, 'reload') as mock_reload:
        response = client.post("/sync?force=true")
        assert response.status_code == 200
        assert response.json()["success"] is True
        state.ds.import_local.assert_called_with(['force'])
        mock_reload.assert_called_once()
