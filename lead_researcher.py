import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
from utils import get_serp_api_key, get_openai_api_key, get_serper_api_key
from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool, ScrapeWebsiteTool
from agents.linkedin_lookup_agent import lookup
from langchain.tools import tool

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4'
os.environ["SERPER_API_KEY"] = get_serper_api_key()

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

#definition of tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
file_read_tool = FileReadTool()
@tool
def linkedin_scrape_tool(lead: str) -> str:
    """
    Finds the LinkedIn URL for the specified company and scrapes its content.

    Args:
        lead (str): The name of the company to search for.

    Returns:
        str: The scraped content from the company's LinkedIn profile.
    """
    try:
        # Step 1: Find the LinkedIn URL using the lookup function
        linkedin_url = lookup(lead)
        if not linkedin_url:
            return f"LinkedIn profile for {lead} not found."

        # Step 2: Scrape the LinkedIn URL
        scraped_content = scrape_tool.run(website_url=linkedin_url)
        return f"Scraped content from LinkedIn profile of {lead}:\n{scraped_content}"
    except Exception as e:
        return f"Error during LinkedIn scraping for {lead}: {str(e)}"


research_task = Task(
    name="Public Information Research Task",
    description=(
        "Conduct a comprehensive public information research task on {lead}. This includes:\n\n"
        "- Collecting general public information about {lead}:\n"
        "- Review {lead}'s official website for details like mission, products/services, leadership, and history.\n"
        "- Identify {lead}'s size.\n"
        "- Identify the {lead}'s industry.\n"
        "- Verifying {lead}'s social media presence:\n"
        "- Analyze activity on LinkedIn. Look for the {lead}'s profile on Linkedin and scrape it. Collect information about its most active period.\n\n"
        "The agent will leverage these tools to deliver a structured and detailed report about {lead}, "
        "enabling stakeholders to make informed decisions based on accurate and comprehensive research."
    ),
expected_output=(
            "A structured report containing the following details:\n"
            "- General public information about {lead}.\n"
            "- Analysis of {lead}'s Linkedin and its engagement at this media."
        ),
    tools=[scrape_tool, linkedin_scrape_tool, file_read_tool],
    agent=lead_research_agent
)

crew = Crew(
    agents = [lead_research_agent],
    tasks = [research_task],
    verbose=2,
    memory=True
)

inputs = {
    "lead": "iFood"
}

result = crew.kickoff(inputs=inputs)



