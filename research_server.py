import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from document_store import DocumentStore
from vector_store import VectorStore
from research_defs import SearchResultEntry, MetadataEntry
from search_tools import SearchTools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("ResearchServer")

class GlobalState:
    def __init__(self):
        self.ds: Optional[DocumentStore] = None
        self.vs: Optional[VectorStore] = None

    def initialize(self):
        log.info("Initializing DocumentStore and VectorStore...")
        self.ds = DocumentStore()
        self.vs = VectorStore(self.ds.storage_path, self.ds.config_path)
        log.info("Initialization complete.")

    def reload(self):
        log.info("Reloading data stores...")
        if self.ds:
            del self.ds
        if self.vs:
            del self.vs
        self.initialize()

state = GlobalState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.initialize()
    yield
    # Cleanup if necessary

app = FastAPI(title="LocalResearch API", lifespan=lifespan)

# --- Pydantic Models ---

class SearchResult(BaseModel):
    id: int
    score: float
    descriptor: str
    text: str | None
    metadata: MetadataEntry | None = None

class TimelineEvent(BaseModel):
    date: str
    event: str

class StatusResponse(BaseModel):
    status: str
    document_count: int
    active_model: str
    models: List[Dict[str, Any]]

class SyncResponse(BaseModel):
    success: bool
    message: str

# --- Endpoints ---

@app.get("/status", response_model=StatusResponse)
async def get_status():
    if not state.ds or not state.vs:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    doc_count = len(state.ds.text_library)
    active_model = state.vs.config.get('embeddings_model_name', 'Unknown')
    
    models = []
    for model in state.vs.model_list:
        models.append({
            "name": model['model_name'],
            "enabled": model['enabled']
        })

    return StatusResponse(
        status="online",
        document_count=doc_count,
        active_model=active_model,
        models=models
    )

@app.post("/sync", response_model=SyncResponse)
async def sync_data(force: bool = False):
    if not state.ds:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    args = []
    if force:
        args.append('force')
    
    # import_local is blocking and uses subprocess, might take a while
    # For a simple local server, blocking the main thread might be acceptable for this operation,
    # but ideally we should run it in a thread.
    # However, since we need to reload the state immediately after, let's keep it simple for now.
    
    try:
        success = state.ds.import_local(args)
        if success:
            state.reload()
            return SyncResponse(success=True, message="Import successful, data reloaded.")
        else:
            return SyncResponse(success=False, message="Import failed or not required.")
    except Exception as e:
        log.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[SearchResult])
async def search(
    q: str, 
    limit: int = 10, 
    highlight: bool = True,
    cutoff: float = 0.0, # Default from console seems to be 0.0 or read from config
    damp: float = 0.0 # Default from console
):
    if not state.vs or not state.ds:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Get defaults from config if not provided (though function args have defaults)
    # In console: 
    # highlight: bool = cast(bool, ds.get_var('highlight', key_vals))
    # cutoff = cast(float, ds.get_var('highlight_cutoff', key_vals))
    
    # We will use the provided args or defaults.
    
    # context_length and context_steps are also used in console, let's add them if needed, 
    # but for now stick to basic search args.
    context_length = 0 # Default
    context_steps = 0 # Default

    try:
        results = state.vs.search(q, state.ds.text_library, limit, highlight, cutoff, damp, context_length, context_steps)
        
        response_data = []
        for i, res in enumerate(results):
            descriptor = res.entry.descriptor
            # Optionally fetch metadata
            metadata = state.ds.get_metadata(descriptor)
            
            response_data.append(SearchResult(
                id=i+1,
                score=res.cosine,
                descriptor=descriptor,
                text=res.text,
                metadata=metadata
            ))
        return response_data
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ksearch", response_model=List[SearchResult])
async def keyword_search(q: str, source: Optional[str] = None):
    if not state.ds:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        results = state.ds.keyword_search(q, source=source)
        
        response_data = []
        # Console reverses the list for display, but API should probably return best first?
        # Console: for index, result in reversed(list(enumerate(search_result_list))):
        # The console reverses it so the best match is at the bottom (closest to prompt).
        # API should return standard order (best match first).
        
        for i, res in enumerate(results):
            descriptor = res.entry.descriptor
            metadata = state.ds.get_metadata(descriptor)
            
            response_data.append(SearchResult(
                id=i+1,
                score=res.cosine, # ksearch returns cosine=1.0 or similar for matches
                descriptor=descriptor,
                text=res.text,
                metadata=metadata
            ))
        return response_data
    except Exception as e:
        log.error(f"Keyword search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeline", response_model=List[TimelineEvent])
async def get_timeline(
    time: Optional[str] = None,
    domains: Optional[str] = None,
    keywords: Optional[str] = None,
    partial_overlap: bool = False,
    full_overlap: bool = False
):
    if not state.ds:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    domain_list = domains.split(' ') if domains else None
    keyword_list = keywords.split(' ') if keywords else None
    
    try:
        tlel = state.ds.tl.search_events(time, domain_list, keyword_list, True, full_overlap, partial_overlap)
        
        events = []
        for tle in tlel:
            date = state.ds.tl.get_date_string_from_event(tle['jd_event'])
            event_text = state.ds.tl.get_event_text(tle['eventdata'])
            if date:
                events.append(TimelineEvent(date=date, event=event_text))
        return events
    except Exception as e:
        log.error(f"Timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{descriptor:path}")
async def get_document(descriptor: str):
    if not state.ds:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    metadata = state.ds.get_metadata(descriptor)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # We could also return the full text content if available in the library
    # But get_metadata returns the metadata dict.
    # To get content, we might need to read the file or check text_library if it's loaded in memory?
    # text_library keys are hashes, not descriptors.
    # But we can get the path.
    
    path = state.ds.get_path_from_descriptor(descriptor)
    content = None
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            content = "Could not read file content."
            
    return {
        "descriptor": descriptor,
        "metadata": metadata,
        "path": path,
        "content": content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
