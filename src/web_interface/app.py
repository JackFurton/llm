import os
import json
import glob
import logging
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('web_interface')

def create_app(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    curated_data_dir: str = "data/curated",
    static_folder: str = None,
    template_folder: str = None
) -> Flask:
    """
    Create a Flask application for the data curation interface
    
    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory containing processed data files
        curated_data_dir: Directory to save curated data files
        static_folder: Path to static files folder
        template_folder: Path to template files folder
        
    Returns:
        Flask application
    """
    # Determine paths for static and template folders
    if static_folder is None:
        static_folder = os.path.join(os.path.dirname(__file__), "static")
    
    if template_folder is None:
        template_folder = os.path.join(os.path.dirname(__file__), "templates")
    
    # Create directories if they don't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(curated_data_dir, exist_ok=True)
    
    # Create Flask app
    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder
    )
    
    # Configure app
    app.config["SECRET_KEY"] = hashlib.sha256(os.urandom(32)).hexdigest()
    app.config["RAW_DATA_DIR"] = os.path.abspath(raw_data_dir)
    app.config["PROCESSED_DATA_DIR"] = os.path.abspath(processed_data_dir)
    app.config["CURATED_DATA_DIR"] = os.path.abspath(curated_data_dir)
    
    # Routes
    @app.route("/")
    def index():
        """Home page"""
        # Count files in each directory
        raw_count = len(glob.glob(os.path.join(app.config["RAW_DATA_DIR"], "*.txt")))
        processed_count = len(glob.glob(os.path.join(app.config["PROCESSED_DATA_DIR"], "*.txt")))
        curated_count = len(glob.glob(os.path.join(app.config["CURATED_DATA_DIR"], "*.txt")))
        
        return render_template(
            "index.html",
            raw_count=raw_count,
            processed_count=processed_count,
            curated_count=curated_count
        )
    
    @app.route("/raw")
    def raw_files():
        """List raw data files"""
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 10, type=int)
        
        # Get all text files
        files = glob.glob(os.path.join(app.config["RAW_DATA_DIR"], "*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Paginate
        total_pages = (len(files) + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get files for current page
        page_files = files[start_idx:end_idx]
        
        # Get file info
        file_info = []
        for file_path in page_files:
            try:
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Get first few lines
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    preview = "".join(f.readline() for _ in range(3))
                
                file_info.append({
                    "filename": filename,
                    "path": file_path,
                    "size": size,
                    "mtime": mtime,
                    "preview": preview
                })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return render_template(
            "file_list.html",
            title="Raw Data Files",
            files=file_info,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            section="raw"
        )
    
    @app.route("/processed")
    def processed_files():
        """List processed data files"""
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 10, type=int)
        
        # Get all text files
        files = glob.glob(os.path.join(app.config["PROCESSED_DATA_DIR"], "*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Paginate
        total_pages = (len(files) + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get files for current page
        page_files = files[start_idx:end_idx]
        
        # Get file info
        file_info = []
        for file_path in page_files:
            try:
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Get first few lines
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    preview = "".join(f.readline() for _ in range(3))
                
                # Check if file is already curated
                curated_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
                is_curated = os.path.exists(curated_path)
                
                file_info.append({
                    "filename": filename,
                    "path": file_path,
                    "size": size,
                    "mtime": mtime,
                    "preview": preview,
                    "is_curated": is_curated
                })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return render_template(
            "file_list.html",
            title="Processed Data Files",
            files=file_info,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            section="processed"
        )
    
    @app.route("/curated")
    def curated_files():
        """List curated data files"""
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 10, type=int)
        
        # Get all text files
        files = glob.glob(os.path.join(app.config["CURATED_DATA_DIR"], "*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Paginate
        total_pages = (len(files) + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get files for current page
        page_files = files[start_idx:end_idx]
        
        # Get file info
        file_info = []
        for file_path in page_files:
            try:
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Get first few lines
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    preview = "".join(f.readline() for _ in range(3))
                
                file_info.append({
                    "filename": filename,
                    "path": file_path,
                    "size": size,
                    "mtime": mtime,
                    "preview": preview
                })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return render_template(
            "file_list.html",
            title="Curated Data Files",
            files=file_info,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            section="curated"
        )
    
    @app.route("/view/<section>/<filename>")
    def view_file(section, filename):
        """View a file"""
        if section == "raw":
            file_path = os.path.join(app.config["RAW_DATA_DIR"], filename)
        elif section == "processed":
            file_path = os.path.join(app.config["PROCESSED_DATA_DIR"], filename)
        elif section == "curated":
            file_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        else:
            abort(404)
        
        # Check if file exists
        if not os.path.exists(file_path):
            abort(404)
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Get file info
            size = os.path.getsize(file_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Check if file is already curated
            curated_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
            is_curated = os.path.exists(curated_path)
            
            return render_template(
                "view_file.html",
                filename=filename,
                content=content,
                size=size,
                mtime=mtime,
                section=section,
                is_curated=is_curated
            )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            flash(f"Error reading file: {e}", "danger")
            return redirect(url_for(f"{section}_files"))
    
    @app.route("/curate/<filename>", methods=["POST"])
    def curate_file(filename):
        """Curate a file"""
        # Get source file path
        source_path = os.path.join(app.config["PROCESSED_DATA_DIR"], filename)
        
        # Check if file exists
        if not os.path.exists(source_path):
            flash(f"File not found: {filename}", "danger")
            return redirect(url_for("processed_files"))
        
        # Get destination file path
        dest_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        
        try:
            # Copy file to curated directory
            shutil.copy2(source_path, dest_path)
            flash(f"File curated: {filename}", "success")
        except Exception as e:
            logger.error(f"Error curating file {filename}: {e}")
            flash(f"Error curating file: {e}", "danger")
        
        # Redirect back to processed files
        return redirect(url_for("processed_files"))
    
    @app.route("/uncurate/<filename>", methods=["POST"])
    def uncurate_file(filename):
        """Remove a file from curated data"""
        # Get file path
        file_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            flash(f"File not found: {filename}", "danger")
            return redirect(url_for("curated_files"))
        
        try:
            # Remove file from curated directory
            os.remove(file_path)
            flash(f"File removed from curated data: {filename}", "success")
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")
            flash(f"Error removing file: {e}", "danger")
        
        # Redirect back to curated files
        return redirect(url_for("curated_files"))
    
    @app.route("/edit/<section>/<filename>", methods=["GET", "POST"])
    def edit_file(section, filename):
        """Edit a file"""
        if section == "raw":
            file_path = os.path.join(app.config["RAW_DATA_DIR"], filename)
        elif section == "processed":
            file_path = os.path.join(app.config["PROCESSED_DATA_DIR"], filename)
        elif section == "curated":
            file_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        else:
            abort(404)
        
        # Check if file exists
        if not os.path.exists(file_path):
            abort(404)
        
        if request.method == "POST":
            try:
                # Get new content
                content = request.form.get("content", "")
                
                # Save file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                flash(f"File saved: {filename}", "success")
                return redirect(url_for(f"{section}_files"))
            except Exception as e:
                logger.error(f"Error saving file {file_path}: {e}")
                flash(f"Error saving file: {e}", "danger")
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            return render_template(
                "edit_file.html",
                filename=filename,
                content=content,
                section=section
            )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            flash(f"Error reading file: {e}", "danger")
            return redirect(url_for(f"{section}_files"))
    
    @app.route("/stats")
    def stats():
        """Show statistics"""
        # Count files in each directory
        raw_count = len(glob.glob(os.path.join(app.config["RAW_DATA_DIR"], "*.txt")))
        processed_count = len(glob.glob(os.path.join(app.config["PROCESSED_DATA_DIR"], "*.txt")))
        curated_count = len(glob.glob(os.path.join(app.config["CURATED_DATA_DIR"], "*.txt")))
        
        # Get total size of each directory
        raw_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(app.config["RAW_DATA_DIR"], "*.txt")))
        processed_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(app.config["PROCESSED_DATA_DIR"], "*.txt")))
        curated_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(app.config["CURATED_DATA_DIR"], "*.txt")))
        
        # Get collection stats if available
        collection_stats_path = os.path.join("data", "collection_stats.json")
        collection_stats = None
        if os.path.exists(collection_stats_path):
            try:
                with open(collection_stats_path, "r") as f:
                    collection_stats = json.load(f)
            except Exception as e:
                logger.error(f"Error reading collection stats: {e}")
        
        # Get preprocessing stats if available
        preprocessing_stats_path = os.path.join("data", "preprocessing_stats.json")
        preprocessing_stats = None
        if os.path.exists(preprocessing_stats_path):
            try:
                with open(preprocessing_stats_path, "r") as f:
                    preprocessing_stats = json.load(f)
            except Exception as e:
                logger.error(f"Error reading preprocessing stats: {e}")
        
        return render_template(
            "stats.html",
            raw_count=raw_count,
            processed_count=processed_count,
            curated_count=curated_count,
            raw_size=raw_size,
            processed_size=processed_size,
            curated_size=curated_size,
            collection_stats=collection_stats,
            preprocessing_stats=preprocessing_stats
        )
    
    @app.route("/api/files/<section>")
    def api_files(section):
        """API endpoint to get files"""
        if section == "raw":
            dir_path = app.config["RAW_DATA_DIR"]
        elif section == "processed":
            dir_path = app.config["PROCESSED_DATA_DIR"]
        elif section == "curated":
            dir_path = app.config["CURATED_DATA_DIR"]
        else:
            return jsonify({"error": "Invalid section"}), 400
        
        # Get all text files
        files = glob.glob(os.path.join(dir_path, "*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Get file info
        file_info = []
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                mtime = os.path.getmtime(file_path)
                
                file_info.append({
                    "filename": filename,
                    "path": file_path,
                    "size": size,
                    "mtime": mtime
                })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return jsonify(file_info)
    
    @app.route("/api/file/<section>/<filename>")
    def api_file(section, filename):
        """API endpoint to get file content"""
        if section == "raw":
            file_path = os.path.join(app.config["RAW_DATA_DIR"], filename)
        elif section == "processed":
            file_path = os.path.join(app.config["PROCESSED_DATA_DIR"], filename)
        elif section == "curated":
            file_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        else:
            return jsonify({"error": "Invalid section"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Get file info
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            
            return jsonify({
                "filename": filename,
                "path": file_path,
                "size": size,
                "mtime": mtime,
                "content": content
            })
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/curate/<filename>", methods=["POST"])
    def api_curate(filename):
        """API endpoint to curate a file"""
        # Get source file path
        source_path = os.path.join(app.config["PROCESSED_DATA_DIR"], filename)
        
        # Check if file exists
        if not os.path.exists(source_path):
            return jsonify({"error": "File not found"}), 404
        
        # Get destination file path
        dest_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        
        try:
            # Copy file to curated directory
            shutil.copy2(source_path, dest_path)
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error curating file {filename}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/uncurate/<filename>", methods=["POST"])
    def api_uncurate(filename):
        """API endpoint to remove a file from curated data"""
        # Get file path
        file_path = os.path.join(app.config["CURATED_DATA_DIR"], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Remove file from curated directory
            os.remove(file_path)
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app

def run_app(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the Flask application
    
    Args:
        host: Host to run the application on
        port: Port to run the application on
        debug: Whether to run in debug mode
        **kwargs: Additional arguments to pass to create_app
    """
    app = create_app(**kwargs)
    app.run(host=host, port=port, debug=debug)
