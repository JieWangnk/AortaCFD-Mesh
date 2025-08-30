"""
CLI entry point for simplified mesh generation package V2
"""

import argparse
import sys
import logging
from pathlib import Path

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mesh_optim.log')
        ]
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AortaCFD Mesh Generation Tool V2 - Simplified Single-Pass Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mesh using simplified approach
  python -m mesh_optim stage1 --geometry cases_input/patient1 --config simple_config_example.json

  # Create configuration template
  python -m mesh_optim create-config --output my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stage 1 subparser
    stage1_parser = subparsers.add_parser('stage1', help='Single-pass mesh generation with ray-casting')
    stage1_parser.add_argument('--geometry', required=True, help='Path to geometry directory (with STL files)')
    stage1_parser.add_argument('--config', help='Configuration file (default: simple_config_example.json)')
    stage1_parser.add_argument('--output', help='Output directory (default: output/<patient>/mesh)')
    stage1_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Config creation subparser
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('--output', default='simple_config.json', help='Output config file name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)
    logger = logging.getLogger('mesh_optim.cli')
    
    try:
        if args.command == 'stage1':
            from .stage1_mesh import Stage1MeshOptimizer
            
            # Use default config if not specified
            if not args.config:
                args.config = Path(__file__).parent.parent / "simple_config_example.json"
            
            # Generate default output path if not specified
            if not args.output:
                patient_name = Path(args.geometry).name
                args.output = Path("output") / patient_name / "mesh_v2"
            
            logger.info(f"🚀 Starting mesh generation V2: {args.geometry}")
            logger.info(f"📁 Output directory: {args.output}")
            logger.info(f"⚙️  Configuration: {args.config}")
            
            optimizer = Stage1MeshOptimizer(args.geometry, args.config, args.output)
            result_dir = optimizer.generate_mesh()
            
            logger.info(f"✅ Mesh generation completed: {result_dir}")
            
        elif args.command == 'create-config':
            # Import the simple mesh generator to create template
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from simple_mesh_generator import create_simple_config_template
            
            create_simple_config_template(Path(args.output))
            logger.info(f"✅ Created configuration template: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())