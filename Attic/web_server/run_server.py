import uvicorn
import logging
from app.config import settings
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn.access") # or "uvicorn" for more general logs
    logger.info(f"Starting Uvicorn server on {settings.server_host}:{settings.server_port}")
    logger.info(f"Serving static files from: {Path(__file__).parent / 'app' / 'static'}")
    logger.info(f"Expecting data resources in: {settings.resources_data_path}")
    
    uvicorn.run(
        "app.main:app", 
        host=settings.server_host, 
        port=settings.server_port, 
        reload=True # Set to False in production
        # workers=settings.server_workers # If you add workers to config
        # ssl_keyfile=settings.tls_keyfile if settings.tls_enabled else None,
        # ssl_certfile=settings.tls_certfile if settings.tls_enabled else None,
    )
