from opentranslator import (
    TranslatorCrew,
    OrganizerAgent,
    SourceCollectorAgent,
    ExecutorAgent,
    ValidatorAgent,
    EditorAgent
)

# Create the translation agents
organizer = OrganizerAgent(
    role="Translation Organizer",
    goal="Efficiently organize and segment translation tasks",
    backstory="Expert at breaking down complex documents while preserving context"
)

source_collector = SourceCollectorAgent(
    role="Resource Collector",
    goal="Gather relevant translation context and resources",
    tools=["GlossaryTool", "TranslationMemoryTool"]
)

executor = ExecutorAgent(
    role="Translation Executor",
    goal="Produce high-quality translations",
    model="gpt-4-turbo",  # Or your preferred translation model
    language_pair=("en", "bn")  # English to Bengali
)

validator = ValidatorAgent(
    role="Quality Validator",
    goal="Ensure translation accuracy and consistency",
    validation_metrics=["BLEU", "BackTranslation"]
)

editor = EditorAgent(
    role="Translation Editor",
    goal="Polish and refine translations",
    language="bn"  # Target language
)

# Create the translation crew
translation_crew = TranslatorCrew(
    agents=[organizer, source_collector, executor, validator, editor],
    verbose=True
)

# Example usage
async def translate_document():
    # Initialize the translation task
    task = {
        "source_text": "path/to/document.pdf",
        "source_language": "en",
        "target_language": "bn",
        "domain": "literary",  # Domain for context
        "style_guide": "path/to/style_guide.yaml"
    }
    
    # Execute the translation
    result = await translation_crew.execute(
        task=task,
        sequential=True,  # Process segments sequentially
        cache_resources=True  # Cache translation memories
    )
    
    # Get the final translation
    translated_document = result.get_translation()
    quality_metrics = result.get_metrics()
    
    return translated_document, quality_metrics

# Run the translation
if __name__ == "__main__":
    import asyncio
    
    document, metrics = asyncio.run(translate_document())
    print(f"Translation completed with quality score: {metrics['overall_score']}")