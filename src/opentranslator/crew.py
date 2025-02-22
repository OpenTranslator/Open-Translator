from opentranslator import Agent, Crew, Process, Task
from opentranslator.project import CrewBase, agent, crew, task
from opentranslator.tools import (
    DocumentAnalyzer,
    GlossaryTool,
    TranslationModel,
    QualityMetricsTool,
    BackTranslationTool
)

@CrewBase
class TranslationCrew:
    """Translation crew for managing the translation pipeline"""

    @agent
    def organizer(self) -> Agent:
        """Creates the Organizer agent"""
        return Agent(
            config=self.agents_config['organizer'],
            verbose=True,
            tools=[DocumentAnalyzer()]
        )

    @agent
    def source_collector(self) -> Agent:
        """Creates the Source Collector agent"""
        return Agent(
            config=self.agents_config['source_collector'],
            verbose=True,
            tools=[GlossaryTool()]
        )

    @agent
    def executor(self) -> Agent:
        """Creates the Executor agent"""
        return Agent(
            config=self.agents_config['executor'],
            verbose=True,
            tools=[TranslationModel()]
        )

    @agent
    def validator(self) -> Agent:
        """Creates the Validator agent"""
        return Agent(
            config=self.agents_config['validator'],
            verbose=True,
            tools=[QualityMetricsTool(), BackTranslationTool()]
        )

    @agent
    def editor(self) -> Agent:
        """Creates the Editor agent"""
        return Agent(
            config=self.agents_config['editor'],
            verbose=True
        )

    @task
    def document_analysis(self) -> Task:
        """Creates the document analysis task"""
        return Task(
            config=self.tasks_config['document_analysis'],
            output_file='analysis.json'
        )

    @task
    def resource_collection(self) -> Task:
        """Creates the resource collection task"""
        return Task(
            config=self.tasks_config['resource_collection'],
            output_file='resources.json'
        )

    @task
    def translation_execution(self) -> Task:
        """Creates the translation execution task"""
        return Task(
            config=self.tasks_config['translation_execution'],
            output_file='translations.json'
        )

    @task
    def quality_validation(self) -> Task:
        """Creates the quality validation task"""
        return Task(
            config=self.tasks_config['quality_validation'],
            output_file='validation.json'
        )

    @task
    def final_editing(self) -> Task:
        """Creates the final editing task"""
        return Task(
            config=self.tasks_config['final_editing'],
            output_file='final_translation.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the translation crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True
        )