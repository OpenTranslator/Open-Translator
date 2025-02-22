#!/usr/bin/env python
import argparse
import os
import sys
from typing import Optional, Dict
from pathlib import Path

from opentranslator.crew import TranslationCrew
from opentranslator.config import load_config

class OpenTranslatorCLI:
    """Command Line Interface for OpenTranslator"""

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description="OpenTranslator - Multi-agent translation framework"
        )
        
        # Main arguments
        parser.add_argument(
            "--task",
            type=str,
            help="Translation task description (e.g., 'Translate the book Great Expectations into Bengali')",
            required=True
        )
        parser.add_argument(
            "--src-lang",
            type=str,
            help="Source language code (e.g., 'en')",
            required=True
        )
        parser.add_argument(
            "--tgt-lang",
            type=str,
            help="Target language code (e.g., 'bn')",
            required=True
        )
        
        # Optional arguments
        parser.add_argument(
            "--input-file",
            type=str,
            help="Path to the input file",
            default=None
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Directory for output files",
            default="./translations"
        )
        parser.add_argument(
            "--domain",
            type=str,
            help="Translation domain (e.g., 'literary', 'technical', 'legal')",
            default="general"
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to custom configuration file",
            default=None
        )
        parser.add_argument(
            "--apikey",
            type=str,
            help="API key for translation service",
            default=None
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        return parser.parse_args()

    @staticmethod
    def setup_environment(args) -> None:
        """Set up environment variables and directories"""
        # Set API key if provided
        if args.apikey:
            os.environ["OPENAI_API_KEY"] = args.apikey
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_task_info(task: str) -> Dict[str, str]:
        """Extract document info from task description"""
        # TODO: Implement more sophisticated task parsing
        return {
            "type": "book" if "book" in task.lower() else "document",
            "title": task.split("'")[1] if "'" in task else None,
            "description": task
        }

    @staticmethod
    def run_translation(args) -> None:
        """Execute the translation task"""
        # Initialize translation crew
        crew = TranslationCrew()
        
        # Prepare inputs
        task_info = OpenTranslatorCLI.extract_task_info(args.task)
        inputs = {
            "source_language": args.src_lang,
            "target_language": args.tgt_lang,
            "domain": args.domain,
            "task_info": task_info,
            "input_file": args.input_file,
            "output_dir": args.output_dir
        }
        
        # Run translation
        try:
            result = crew.crew().kickoff(inputs=inputs)
            print(f"\n✅ Translation completed successfully!")
            print(f"📁 Output saved to: {result.output_path}")
        except Exception as e:
            print(f"\n❌ Error during translation: {str(e)}")
            sys.exit(1)

def main():
    """Main entry point for the CLI"""
    # Parse command line arguments
    args = OpenTranslatorCLI.parse_args()
    
    try:
        # Setup environment
        OpenTranslatorCLI.setup_environment(args)
        
        # Validate required environment variables
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please provide it using --apikey or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Run translation
        OpenTranslatorCLI.run_translation(args)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()