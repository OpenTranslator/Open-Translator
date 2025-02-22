#!/usr/bin/env python
from opentranslator.crew import TranslationCrew
from opentranslator.config import load_config

def run():
    """
    Run the translation crew with configuration.
    """
    inputs = {
        "source_language": "en",
        "target_language": "bn",
        "domain": "literary",
        "document_path": "path/to/your/document.pdf"
    }
    
    # Initialize and run the translation crew
    translation_crew = TranslationCrew()
    result = translation_crew.crew().kickoff(inputs=inputs)
    
    print(f"Translation completed. Output saved to: {result.output_path}")

if __name__ == "__main__":
    run()