import argparse
from pathlib import Path
from typing import Optional

def download_model(model_name: str = 'yolo11s', output_dir: str = './models', format_type: str = 'pt') -> Optional[Path]:
    """
    Download a YOLO model using ultralytics and save it to the specified directory.
    
    Args:
        model_name: Name of the YOLO model to download (default: yolo11s)
        output_dir: Directory to save the model
    
    Returns:
        Path to the downloaded model file or None if download failed
    """
    # Create output directory if it doesn't exist
    model_dir = Path(output_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    file_ext = f".{format_type}" if not format_type.startswith('.') else format_type
    model_path = model_dir / f"{model_name}{file_ext}"
    
    # If model already exists, return its path
    if model_path.exists():
        print(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    print(f"Downloading {model_name} model...")
    try:
        # Import ultralytics YOLO
        from ultralytics import YOLO
        
        # Download the model - ultralytics will handle the downloading
        model = YOLO(model_dir / f"{model_name}.pt")

        # Save in the requested format
        if format_type.lower() == 'pt':
            # Save as PyTorch model
            model.save(model_path)
            print(f"Model downloaded and saved to {model_path}")
            return model_path
        elif format_type.lower() == 'onnx':
            # Export as ONNX model
            export_path = model.export(format='onnx', imgsz=640)
            model.save(export_path)
            
            # Move the exported model to the desired location if needed
            if Path(export_path) != model_path:
                import shutil
                shutil.move(export_path, model_path)
                
            print(f"Model downloaded and exported to ONNX format at {model_path}")
            return model_path
        else:
            print(f"Unsupported format: {format_type}. Using default PyTorch format.")
            model.save(model_dir / f"{model_name}.pt")
            return model_dir / f"{model_name}.pt"
    
    except FileNotFoundError:
        print(f"Error: Model {model_name} not found. Please check the model name.")
        return None
    
    except ImportError:
        print("Error: ultralytics package not installed. Install with: pip install ultralytics")
        return None
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def main() -> None:
    """Parse command-line arguments and download the model."""
    parser = argparse.ArgumentParser(description='Download YOLO model')
    parser.add_argument('--model', default='yolo11s', help='Model name (default: yolo11s)')
    parser.add_argument('--output', default='./models', help='Output directory (default: ./models)')
    parser.add_argument('--format', default='pt', choices=['pt', 'onnx'], help='Model format (default: pt)')
    
    args = parser.parse_args()
    model_path = download_model(args.model, args.output, args.format)
    
    if model_path:
        print(f"Model setup complete: {model_path}")
    else:
        print("Model setup failed. Please check error messages above.")

if __name__ == "__main__":
    main()