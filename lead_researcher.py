import warnings
from crewai import Agent, Task, Crew
import os
from utils import get_serp_api_key, get_openai_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

lead_research_agent = Agent(
    role="Lead Researcher",
    goal="Develop a study about a potential lead: {lead}. ",
    backstory="You are working for a company that is willing to find"
             "new leads. Your job is to collect and verify public information about the company {lead}, "
             "including its website, social media presence, and historical interactions."
             "With these information on hand, your team will be able to dig deep into the opportunities"
             "working with that lead",
allow_delegation=False,
verbose=True

)

research_task = Task(
    name="Public Information Research Task",
    description=(
        "Conduct a comprehensive public information research task on {lead}. This includes:\n\n"
        "- Collecting general public information about {lead}:\n"
        "  - Review {lead}'s official website for details like mission, products/services, leadership, and history.\n"
        "  - Identify {lead}'s size, industry focus, and notable achievements or partnerships.\n\n"
        "- Verifying {lead}'s social media presence:\n"
        "  - Analyze activity on platforms such as LinkedIn, Twitter (X), Facebook, Instagram, and YouTube.\n"
        "  - Look for customer engagement, sentiment, and recent campaigns.\n\n"
        "- Gathering {lead}'s interaction history:\n"
        "  - Search for records of previous dealings, reviews, or interactions with stakeholders.\n"
        "  - Highlight any partnerships, disputes, or collaborations in {lead}'s history.\n\n"
        "The agent will leverage these tools to deliver a structured and detailed report about {lead}, "
        "enabling stakeholders to make informed decisions based on accurate and comprehensive research."
    ),
    tools=[],
    agent=lead_research_agent
)
