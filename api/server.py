"""
FastAPI server for Paper2Slides
"""

import sys
import uuid
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
DEFAULT_CONTENT_TYPE = "general"
DEFAULT_OUTPUT_TYPE = "slides"
DEFAULT_STYLE = "Executive consulting deck"
DEFAULT_SLIDES_LENGTH = "long"
DEFAULT_POSTER_DENSITY = "medium"
DEFAULT_PRESENTATION_PROFILE = "consulting_exec_cn"
SUPPORTED_SOURCE_EXTENSIONS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
}

# Add parent directory to path to import paper2slides modules
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "paper2slides" / ".env"):
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        break

DEFAULT_PRESENTATION_PROFILE = os.getenv("PAPER2SLIDES_PROFILE", DEFAULT_PRESENTATION_PROFILE).strip() or DEFAULT_PRESENTATION_PROFILE

# Import paper2slides functions
from paper2slides.core import (
    run_pipeline, get_base_dir, get_config_dir,
    get_config_name, detect_start_stage
)
from paper2slides.utils.path_utils import get_project_name
from paper2slides.utils import setup_logging, save_json, load_json

# Configuration - use project root directories
UPLOAD_DIR = PROJECT_ROOT / "sources" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Paper2Slides API", version="1.0.0")

# Configure logging for paper2slides
setup_logging(level=logging.INFO)

# Global state for tracking running sessions
class SessionManager:
    def __init__(self):
        self.running_session = None
        self.cancelled_sessions = set()  # Track cancelled session IDs
        self.lock = threading.Lock()
    
    async def start_session(self, session_id: str) -> bool:
        """Try to start a new session. Returns False if another session is already running"""
        with self.lock:
            if self.running_session is not None:
                return False
            self.running_session = session_id
            # Remove from cancelled set when starting (for regeneration cases)
            self.cancelled_sessions.discard(session_id)
            return True
    
    async def end_session(self, session_id: str):
        """End a session"""
        with self.lock:
            if self.running_session == session_id:
                self.running_session = None
            # Keep cancelled flag for a bit, clean up later if needed
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session. Returns True if session was running"""
        with self.lock:
            if self.running_session == session_id:
                self.cancelled_sessions.add(session_id)
                logger.info(f"Session {session_id[:8]} marked for cancellation")
                return True
            return False
    
    def is_cancelled(self, session_id: str) -> bool:
        """Check if a session has been cancelled"""
        return session_id in self.cancelled_sessions
    
    def get_running_session(self) -> Optional[str]:
        """Get the currently running session ID"""
        return self.running_session

session_manager = SessionManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving generated files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
# Mount uploads directory for serving uploaded source files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


class ChatResponse(BaseModel):
    message: str
    slides: Optional[List[dict]] = None
    ppt_url: Optional[str] = None
    poster_url: Optional[str] = None
    session_id: Optional[str] = None
    uploaded_files: Optional[List[dict]] = None


class PresentationJobResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str
    cancel_url: str
    uploaded_files: List[dict]


async def require_api_auth(authorization: Optional[str] = Header(None)):
    """Protect external API endpoints when API_AUTH_TOKEN is configured."""
    if not API_AUTH_TOKEN:
        return
    if authorization != f"Bearer {API_AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _public_url(path: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{path}"
    return path


def _job_result_path(session_id: str) -> Path:
    return UPLOAD_DIR / session_id / "result.json"


def _persist_job_result(session_id: str, result: dict):
    result_path = _job_result_path(session_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result_path, result)


def _load_job_result(session_id: str) -> Optional[dict]:
    if hasattr(app.state, 'results') and session_id in app.state.results:
        return app.state.results[session_id]
    return load_json(_job_result_path(session_id))


def _validate_api_choice(value: str, allowed: set[str], field_name: str):
    if value not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: {value}. Allowed values: {', '.join(sorted(allowed))}",
        )


def _normalize_profile(profile: Optional[str]) -> str:
    """Normalize empty profile input to the consulting default."""
    return str(profile or "").strip() or DEFAULT_PRESENTATION_PROFILE


def _normalize_style_request(style: Optional[str], message: str = "") -> tuple[str, Optional[str]]:
    """Normalize incoming style to generator style/custom_style values."""
    predefined_styles = {"academic", "doraemon"}

    if message and message.strip():
        return "custom", message.strip()

    normalized = str(style or "").strip()
    lowered = normalized.lower()
    if lowered in predefined_styles:
        return lowered, None
    if lowered == "consulting" or not normalized:
        return "custom", DEFAULT_STYLE
    return "custom", normalized


@app.get("/")
async def root():
    return {"message": "Paper2Slides API Server", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/v1/capabilities", dependencies=[Depends(require_api_auth)])
async def get_capabilities():
    """Describe the stable service API surface for external callers."""
    return {
        "service": "paper2slides",
        "supported_source_extensions": sorted(SUPPORTED_SOURCE_EXTENSIONS),
        "content_types": ["paper", "general"],
        "output_types": ["slides", "poster"],
        "styles": ["consulting", "academic", "doraemon", "custom"],
        "profiles": [DEFAULT_PRESENTATION_PROFILE],
        "default_profile": DEFAULT_PRESENTATION_PROFILE,
        "slides_lengths": ["short", "medium", "long"],
        "poster_densities": ["sparse", "medium", "dense"],
        "artifacts": ["slides.pdf", "slide images", "poster image"],
    }


async def _save_uploaded_files(session_id: str, files: List[UploadFile]) -> List[dict]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one source file is required")

    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    saved_files = []
    unsupported = []

    for file in files:
        if not file.filename:
            continue

        original_name = Path(file.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_SOURCE_EXTENSIONS:
            unsupported.append(original_name)
            continue

        file_path = session_dir / original_name
        chunk_size = 1024 * 1024
        with open(file_path, "wb") as buffer:
            while chunk := file.file.read(chunk_size):
                buffer.write(chunk)

        saved_files.append({
            "filename": original_name,
            "path": str(file_path),
            "size": file_path.stat().st_size,
        })
        logger.info(f"Saved service upload: {file_path}")

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported source file type",
                "unsupported_files": unsupported,
                "supported_extensions": sorted(SUPPORTED_SOURCE_EXTENSIONS),
            },
        )

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid source files were uploaded")

    return saved_files


@app.post("/api/v1/presentations", response_model=PresentationJobResponse, dependencies=[Depends(require_api_auth)])
async def create_presentation_job(
    background_tasks: BackgroundTasks,
    content: str = Form(DEFAULT_CONTENT_TYPE),
    output_type: str = Form(DEFAULT_OUTPUT_TYPE),
    style: str = Form(DEFAULT_STYLE),
    style_prompt: str = Form(""),
    profile: str = Form(DEFAULT_PRESENTATION_PROFILE),
    length: Optional[str] = Form(DEFAULT_SLIDES_LENGTH),
    density: Optional[str] = Form(DEFAULT_POSTER_DENSITY),
    fast_mode: bool = Form(False),
    files: List[UploadFile] = File(...),
):
    """
    Stable external integration endpoint.

    It accepts source documents, starts the pipeline asynchronously, and returns
    a job id for polling status and downloading generated artifacts.
    """
    _validate_api_choice(content, {"paper", "general"}, "content")
    _validate_api_choice(output_type, {"slides", "poster"}, "output_type")
    if length is not None:
        _validate_api_choice(length, {"short", "medium", "long"}, "length")
    if density is not None:
        _validate_api_choice(density, {"sparse", "medium", "dense"}, "density")

    running_session = session_manager.get_running_session()
    if running_session:
        raise HTTPException(
            status_code=409,
            detail=f"Another generation job is already running: {running_session[:8]}",
        )

    session_id = str(uuid.uuid4())
    saved_files = await _save_uploaded_files(session_id, files)
    _persist_job_result(session_id, {"status": "queued"})

    background_tasks.add_task(
        run_pipeline_background_sync,
        session_id,
        style_prompt,
        _normalize_profile(profile),
        saved_files,
        content,
        output_type,
        style,
        length,
        density,
        fast_mode,
        session_manager,
    )

    return {
        "job_id": session_id,
        "status": "queued",
        "status_url": _public_url(f"/api/v1/presentations/{session_id}"),
        "result_url": _public_url(f"/api/v1/presentations/{session_id}/result"),
        "cancel_url": _public_url(f"/api/v1/presentations/{session_id}/cancel"),
        "uploaded_files": [
            {
                "name": f["filename"],
                "size": f["size"],
                "url": _public_url(f"/uploads/{session_id}/{quote(f['filename'])}"),
            }
            for f in saved_files
        ],
    }


@app.get("/api/session/running")
async def get_running_session():
    """Check if there is a session currently running"""
    running_session = session_manager.get_running_session()
    return {
        "has_running_session": running_session is not None,
        "running_session_id": running_session[:8] if running_session else None
    }


@app.post("/api/cancel/{session_id}")
async def cancel_session(session_id: str):
    """Cancel a running session"""
    try:
        cancelled = await session_manager.cancel_session(session_id)
        if cancelled:
            return {"message": f"Session {session_id[:8]} cancellation requested", "cancelled": True}
        else:
            return {"message": f"Session {session_id[:8]} is not running", "cancelled": False}
    except Exception as e:
        logger.error(f"Error cancelling session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/presentations/{job_id}/cancel", dependencies=[Depends(require_api_auth)])
async def cancel_presentation_job(job_id: str):
    """Cancel a running external presentation generation job."""
    return await cancel_session(job_id)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    content: str = Form(DEFAULT_CONTENT_TYPE),  # 'paper' or 'general'
    output_type: str = Form(DEFAULT_OUTPUT_TYPE),  # 'slides' or 'poster'
    style: str = Form(DEFAULT_STYLE),  # 'academic', 'doraemon', or custom description
    profile: str = Form(DEFAULT_PRESENTATION_PROFILE),  # unified prompt profile
    length: Optional[str] = Form(None),  # 'short', 'medium', 'long' (for slides)
    density: Optional[str] = Form(None),  # 'sparse', 'medium', 'dense' (for poster)
    fast_mode: Optional[str] = Form(None),  # 'true' or 'false' - fast mode for paper content
    session_id: Optional[str] = Form(None),  # Existing session ID to reuse files
    files: List[UploadFile] = File([])
):
    """
    Main chat endpoint that receives files and instructions
    
    Args:
        message: User's text message
        content: 'paper' or 'general'
        output_type: 'slides' or 'poster'
        style: 'academic', 'doraemon', or custom description
        length: 'short', 'medium', 'long' (for slides)
        density: 'sparse', 'medium', 'dense' (for poster)
        fast_mode: 'true' or 'false' - fast mode for paper content (no RAG indexing)
        session_id: Optional existing session ID to reuse files (for regeneration)
        files: List of uploaded files (PDF, MD, etc.)
    
    Returns:
        Response with session ID - actual generation happens in background
    """
    try:
        # Check if another session is already running
        running_session = session_manager.get_running_session()
        
        # Check if reusing existing session
        reusing_session = False
        if session_id and not files:
            # Reuse existing session
            session_dir = UPLOAD_DIR / session_id
            if session_dir.exists():
                reusing_session = True
                print(f"Reusing existing session: {session_id[:8]}")
                
                # Check if this is a different session from the running one
                if running_session and running_session != session_id:
                    raise HTTPException(
                        status_code=409, 
                        detail=f"Another session is already running. Please wait for it to complete. Running session: {running_session[:8]}"
                    )
            else:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        else:
            # Generate new session ID
            if running_session:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Another session is already running. Please wait for it to complete. Running session: {running_session[:8]}"
                )
            
            session_id = str(uuid.uuid4())
            session_dir = UPLOAD_DIR / session_id
            session_dir.mkdir(exist_ok=True)
        
        # Save uploaded files or load existing files
        saved_files = []
        if reusing_session:
            # Load existing files from session directory
            for file_path in session_dir.iterdir():
                if file_path.is_file():
                    saved_files.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size
                    })
            print(f"Loaded {len(saved_files)} existing file(s) from session")
        else:
            # Save newly uploaded files
            for file in files:
                if file.filename:
                    file_path = session_dir / file.filename
                    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                    with open(file_path, "wb") as buffer:
                        while chunk := file.file.read(CHUNK_SIZE):
                            buffer.write(chunk)
                    saved_files.append({
                        "filename": file.filename,
                        "path": str(file_path),
                        "size": file_path.stat().st_size
                    })
                    print(f"Saved file: {file_path}")
        
        # Parse fast_mode from string to boolean
        fast_mode_bool = fast_mode and fast_mode.lower() == 'true'
        
        # Log received request
        print(f"\n{'='*60}")
        print(f"New Request (Session: {session_id[:8]})")
        print(f"Files: {len(saved_files)} file(s)")
        for f in saved_files:
            print(f"  - {f['filename']} ({f['size']} bytes)")
        print(f"Config: {output_type} | {style} | {content}")
        print(f"  Profile: {_normalize_profile(profile)}")
        if length:
            print(f"  Length: {length}")
        if density:
            print(f"  Density: {density}")
        if content == 'paper' and fast_mode_bool:
            print(f"  Fast Mode: enabled")
        print(f"{'='*60}\n")
        
        # Prepare initial response with session_id and uploaded files
        response_data = {
            "message": f"Processing {len(saved_files)} file(s)...",
            "session_id": session_id,
            "uploaded_files": [
                {
                    "name": f["filename"],
                    "size": f["size"],
                    "url": f"/uploads/{session_id}/{f['filename']}"
                }
                for f in saved_files
            ],
            "slides": [],
            "ppt_url": None,
            "poster_url": None
        }
        
        # Start the pipeline in background
        background_tasks.add_task(
            run_pipeline_background_sync,
            session_id,
            message,
            _normalize_profile(profile),
            saved_files,
            content,
            output_type,
            style,
            length,
            density,
            fast_mode_bool,
            session_manager  # Pass session manager to check for cancellation
        )
        
        # Return immediately so frontend can start polling
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


def _get_source_files(files: List[dict]) -> List[dict]:
    """Return uploaded files that are valid source documents."""
    return [
        f for f in files
        if Path(f["filename"]).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
    ]


async def generate_slides_with_pipeline(
    session_id: str,
    message: str, 
    profile: str,
    files: List[dict], 
    content: str, 
    output_type: str, 
    style: str,
    length: Optional[str] = None,
    density: Optional[str] = None,
    fast_mode: bool = False,
    session_manager: SessionManager = None
) -> dict:
    """
    Run the actual Paper2Slides pipeline
    
    Args:
        session_id: Unique session ID for this upload
        message: User message
        files: List of saved file info
        content: 'paper' or 'general'
        output_type: 'slides' or 'poster'
        style: 'academic', 'doraemon', or custom description
        length: 'short', 'medium', 'long' (for slides)
        density: 'sparse', 'medium', 'dense' (for poster)
        fast_mode: Fast mode for paper content (no RAG indexing)
    
    Returns:
        Dictionary with slides info and output paths
    """
    source_files = _get_source_files(files)
    if not source_files:
        raise ValueError("No supported source file found in uploaded files")
    
    style_type, custom_style = _normalize_style_request(style, message)
    normalized_profile = _normalize_profile(profile)
    
    source_paths = [f['path'] for f in source_files]
    
    # Determine paths (using session-based directory for multiple PDFs)
    if len(source_paths) > 1:
        # Multiple PDFs: use session_id as the identifier
        project_name = f"session_{session_id[:8]}"
        # Use session directory as input_path for multiple files
        input_path = str(Path(source_paths[0]).parent)
        print(f"Processing {len(source_paths)} source files as a single project")
    else:
        project_name = get_project_name(source_paths[0])
        input_path = source_paths[0]
    
    # Build config matching main.py format
    config = {
        "input_path": input_path,  # Required by RAG stage
        "source_paths": source_paths,
        "content_type": content,
        "output_type": output_type,
        "style": style_type,
        "custom_style": custom_style,
        "slides_length": length or "medium",
        "poster_density": density or "medium",
        "fast_mode": fast_mode if content == "paper" else False,  # Fast mode only for paper content
        "output_language": os.getenv("OUTPUT_LANGUAGE", "zh-CN"),
        "profile": normalized_profile,
    }
    
    base_dir = get_base_dir(str(OUTPUT_DIR), project_name, content)
    config_dir = get_config_dir(base_dir, config)
    
    print(f"\nPipeline Configuration:")
    print(f"  Project: {project_name}")
    print(f"  Files: {len(source_paths)}")
    for i, path in enumerate(source_paths, 1):
        print(f"    [{i}] {Path(path).name}")
    if message and message.strip():
        print(f"  Message: {message}")
    print(f"  Profile: {normalized_profile}")
    print(f"  Output: {base_dir}")
    print(f"  Config: {config_dir.name}")
    
    # Detect start stage first
    from_stage = detect_start_stage(base_dir, config_dir, config)
    print(f"Starting from stage: {from_stage}")
    
    # Create initial state BEFORE starting pipeline
    from paper2slides.core.state import create_state, save_state, STAGES
    initial_state = create_state(config)
    
    # Add session_id to state for tracking
    initial_state["session_id"] = session_id
    
    # Mark stages before from_stage as completed (they are being reused)
    start_idx = STAGES.index(from_stage)
    for i in range(start_idx):
        initial_state["stages"][STAGES[i]] = "completed"
    
    save_state(config_dir, initial_state)
    print(f"  Initial state saved (starting from {from_stage})")
    
    # Run the pipeline (base_dir already handles document grouping)
    # Pass session_manager to enable cancellation checks
    await run_pipeline(base_dir, config_dir, config, from_stage, session_id, session_manager)
    
    # Find generated output
    output_files = []
    if config_dir.exists():
        # Find latest timestamped directory
        timestamp_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamp_dirs:
            latest_output = timestamp_dirs[0]
            # Collect generated files
            for file_path in latest_output.iterdir():
                if file_path.is_file():
                    output_files.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(OUTPUT_DIR))
                    })
    
    return {
        "output_dir": str(config_dir),
        "output_files": output_files,
        "num_files": len(output_files)
    }


def _update_state_on_error(
    session_id: str,
    error_msg: str,
    files: List[dict],
    profile: str,
    content: str,
    output_type: str,
    style: str,
    length: Optional[str],
    density: Optional[str],
    fast_mode: bool
):
    """Update state.json when background pipeline fails"""
    from paper2slides.core.state import load_state, save_state
    import json
    
    source_files = _get_source_files(files)
    if not source_files:
        return
    
    source_paths = [f['path'] for f in source_files]
    if len(source_paths) > 1:
        project_name = f"session_{session_id[:8]}"
    else:
        project_name = get_project_name(source_paths[0])
    
    # Determine which stage failed by checking state file
    base_dir = get_base_dir(str(OUTPUT_DIR), project_name, content)
    
    # Build config to find config_dir
    style_type, custom_style = _normalize_style_request(style)
    
    config = {
        "output_type": output_type,
        "style": style_type,
        "custom_style": custom_style,
        "slides_length": length or "medium",
        "poster_density": density or "medium",
        "fast_mode": fast_mode if content == "paper" else False,
        "output_language": os.getenv("OUTPUT_LANGUAGE", "zh-CN"),
        "profile": _normalize_profile(profile),
    }
    
    config_dir = get_config_dir(base_dir, config)
    
    # Load and update state
    state = load_state(config_dir)
    if state:
        # Find the running stage and mark it as failed
        for stage_name, stage_status in state.get("stages", {}).items():
            if stage_status == "running":
                state["stages"][stage_name] = "failed"
                break
        state["error"] = error_msg
        save_state(config_dir, state)
        logger.info(f"Updated state.json with error for session {session_id[:8]}")


async def run_pipeline_background(
    session_id: str,
    message: str,
    profile: str,
    files: List[dict],
    content: str,
    output_type: str,
    style: str,
    length: Optional[str],
    density: Optional[str],
    fast_mode: bool = False,
    session_manager: SessionManager = None
):
    """
    Run pipeline in background and store results
    """
    try:
        # Try to start the session
        can_start = await session_manager.start_session(session_id)
        if not can_start:
            logger.error(f"Cannot start session {session_id[:8]} - another session is already running")
            # Store error in state
            if not hasattr(app.state, 'results'):
                app.state.results = {}
            app.state.results[session_id] = {"error": "Another session is already running"}
            return
        
        logger.info(f"Starting background pipeline for session {session_id[:8]}")
        result = await generate_slides_with_pipeline(
            session_id, message, profile, files, content, output_type, style, length, density, fast_mode, session_manager
        )
        
        # Check if cancelled after completion
        if session_manager and session_manager.is_cancelled(session_id):
            logger.info(f"Session {session_id[:8]} was cancelled")
            raise Exception("Generation cancelled by user")
        
        logger.info(f"Background pipeline completed for session {session_id[:8]}")
        
        # Store result in a simple cache (in production, use Redis or database)
        if not hasattr(app.state, 'results'):
            app.state.results = {}
        app.state.results[session_id] = result
        _persist_job_result(session_id, result)
        
    except Exception as e:
        logger.error(f"Background pipeline failed for session {session_id[:8]}: {e}", exc_info=True)
        # Store error in state
        if not hasattr(app.state, 'results'):
            app.state.results = {}
        app.state.results[session_id] = {"error": str(e)}
        _persist_job_result(session_id, {"error": str(e)})
        
        # Also update the state.json file to reflect the failure
        try:
            _update_state_on_error(session_id, str(e), files, profile, content, output_type, style, length, density, fast_mode)
        except Exception as state_err:
            logger.error(f"Failed to update state file: {state_err}")
    finally:
        # Always end the session when done (success or failure)
        await session_manager.end_session(session_id)
        logger.info(f"Session {session_id[:8]} ended")


def run_pipeline_background_sync(*args, **kwargs):
    """Run async pipeline work in a worker thread so FastAPI stays responsive."""
    asyncio.run(run_pipeline_background(*args, **kwargs))


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Get processing status for a session"""
    try:
        # Find the output directory for this session
        session_dir = UPLOAD_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        source_files = [p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS]
        if not source_files:
            return {"session_id": session_id, "status": "no_files", "stages": {}}
        
        # Determine project name and paths
        if len(source_files) > 1:
            project_name = f"session_{session_id[:8]}"
        else:
            project_name = get_project_name(str(source_files[0]))
        
        # Try to find the state file matching session_id
        from paper2slides.core.paths import get_base_dir
        import json
        
        # Check both paper and general content types
        state_data = None
        most_recent_time = None
        
        for content_type in ["paper", "general"]:
            base_dir = Path(get_base_dir(str(OUTPUT_DIR), project_name, content_type))
            if base_dir.exists():
                # Look for all state.json files in config directories
                for state_file_path in base_dir.rglob("state.json"):
                    if state_file_path.is_file():
                        try:
                            with open(state_file_path, 'r') as f:
                                current_state = json.load(f)
                            
                            # First priority: exact match by session_id
                            if current_state.get("session_id") == session_id:
                                state_data = current_state
                                logger.debug(f"Found exact session match: {state_file_path}")
                                break
                            
                            # Second priority: most recently updated (fallback for old state files)
                            updated_at = current_state.get("updated_at") or current_state.get("created_at")
                            if updated_at:
                                if most_recent_time is None or updated_at > most_recent_time:
                                    most_recent_time = updated_at
                                    # Only use as fallback if no exact match found
                                    if state_data is None or state_data.get("session_id") != session_id:
                                        state_data = current_state
                        except Exception as e:
                            logger.warning(f"Error reading state file {state_file_path}: {e}")
                            continue
                
                # If found exact match, stop searching
                if state_data and state_data.get("session_id") == session_id:
                    break
        
        if not state_data:
            return {
                "session_id": session_id,
                "status": "pending",
                "stages": {
                    "rag": "pending",
                    "summary": "pending",
                    "plan": "pending",
                    "generate": "pending"
                }
            }
        
        # Determine overall status
        stages = state_data.get("stages", {})
        if any(status == "failed" for status in stages.values()):
            overall_status = "failed"
        elif all(status == "completed" for status in stages.values()):
            overall_status = "completed"
        elif any(status == "running" for status in stages.values()):
            overall_status = "running"
        else:
            overall_status = "pending"

        return {
            "session_id": session_id,
            "status": overall_status,
            "stages": stages,
            "error": state_data.get("error"),
            "updated_at": state_data.get("updated_at")
        }
        
    except Exception as e:
        logger.error(f"Error getting status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/presentations/{job_id}", dependencies=[Depends(require_api_auth)])
async def get_presentation_job_status(job_id: str):
    """Get external presentation generation job status."""
    return await get_status(job_id)


def _build_external_result(session_id: str, result: dict) -> dict:
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    artifacts = []
    primary_artifact = None

    for output_file in result.get("output_files", []):
        filename = output_file["filename"]
        relative_path = output_file["relative_path"]
        download_path = f"/api/download/{quote(relative_path, safe='/')}"

        if filename == "slides.pdf":
            artifact_type = "slides_pdf"
        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            artifact_type = "image"
        else:
            artifact_type = "file"

        artifact = {
            "filename": filename,
            "type": artifact_type,
            "url": _public_url(download_path),
            "preview_url": _public_url(f"/outputs/{quote(relative_path, safe='/')}"),
        }
        artifacts.append(artifact)

        if primary_artifact is None or artifact_type == "slides_pdf":
            primary_artifact = artifact

    return {
        "job_id": session_id,
        "status": "completed",
        "output_dir": result.get("output_dir"),
        "num_files": result.get("num_files", len(artifacts)),
        "primary_artifact": primary_artifact,
        "artifacts": artifacts,
    }


@app.get("/api/v1/presentations/{job_id}/result", dependencies=[Depends(require_api_auth)])
async def get_presentation_job_result(job_id: str):
    """Get generated artifacts for an external presentation job."""
    result = _load_job_result(job_id)
    if not result:
        return JSONResponse(
            status_code=202,
            content={"message": "Result not ready yet", "job_id": job_id},
        )
    if result.get("status") == "queued":
        return JSONResponse(
            status_code=202,
            content={"message": "Result not ready yet", "job_id": job_id, "status": "queued"},
        )
    return _build_external_result(job_id, result)


@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    """Get the final result for a completed session"""
    try:
        # Check if result is in cache
        result = _load_job_result(session_id)
        if result:
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Prepare response with output files
            output_files = result.get("output_files", [])
            
            # Find PDF file in output
            pdf_file = next((f for f in output_files if f['filename'].endswith('.pdf')), None)
            # Find image files
            image_files = [f for f in output_files if f['filename'].endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Get output_type from state
            session_dir = UPLOAD_DIR / session_id
            source_files = [p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS]
            if len(source_files) > 1:
                project_name = f"session_{session_id[:8]}"
            else:
                project_name = get_project_name(str(source_files[0]))
            
            # Try to get output type from state
            from paper2slides.core.paths import get_base_dir
            import json
            
            output_type = "slides"  # default
            for content_type in ["paper", "general"]:
                base_dir = Path(get_base_dir(str(OUTPUT_DIR), project_name, content_type))
                if base_dir.exists():
                    for state_file_path in base_dir.rglob("state.json"):
                        if state_file_path.is_file():
                            try:
                                with open(state_file_path, 'r') as f:
                                    state_data = json.load(f)
                                    output_type = state_data.get("config", {}).get("output_type", "slides")
                                    break
                            except:
                                pass
                    if output_type != "slides":
                        break
            
            response_data = {
                "session_id": session_id,
                "slides": [
                    {
                        "title": f"Slide {i+1}",
                        "image_url": f"/outputs/{img['relative_path']}"
                    }
                    for i, img in enumerate(image_files)
                ],
            }
            
            # Add download links
            if pdf_file:
                if output_type == "slides":
                    response_data["ppt_url"] = f"/outputs/{pdf_file['relative_path']}"
                elif output_type == "poster":
                    response_data["poster_url"] = f"/outputs/{pdf_file['relative_path']}"
            elif image_files and output_type == "poster":
                # If no PDF but has images, use first image as poster
                response_data["poster_url"] = f"/outputs/{image_files[0]['relative_path']}"
            
            return JSONResponse(content=response_data)
        
        # If not in cache, return not ready
        return JSONResponse(
            status_code=202,
            content={"message": "Result not ready yet", "session_id": session_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting result for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filepath:path}")
async def download_file(filepath: str):
    """Download generated file (supports subdirectories)"""
    file_path = OUTPUT_DIR / filepath
    
    # Security check: ensure file is within OUTPUT_DIR
    try:
        file_path = file_path.resolve()
        OUTPUT_DIR.resolve()
        if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Allow port to be specified via command line argument
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    
    print("Starting Paper2Slides API server...")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Server running on http://0.0.0.0:{port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300,  
        limit_concurrency=10,    
        limit_max_requests=1000  
    )
