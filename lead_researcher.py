import warnings

from tensorflow_datasets.d4rl.dataset_utils import description

warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
from utils import get_serp_api_key, get_openai_api_key, get_serper_api_key
from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool, ScrapeWebsiteTool
from agents.linkedin_lookup_agent import linkedin_lookup
from agents.twitter_lookup_agent import twitter_lookup
from langchain.tools import tool

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4'
os.environ["SERPER_API_KEY"] = get_serper_api_key()

lead_research_agent = Agent(
    role="Lead Researcher",
    goal="Develop a study about a potential lead: {lead}. ",
    backstory="You are working for a company that is willing to find"
             "new leads. You work at goMerlin: a Brazilian technology company specializing in workforce management and optimization solutions for businesses. "
             "Their primary focus is on leveraging cutting-edge technology to enhance operational efficiency and streamline employee scheduling processes."
             "With a comprehensive suite of tools, goMerlin offers tailored solutions to meet the dynamic needs of companies in various industries."
             "Their platform is designed for businesses seeking to improve workforce management, enhance productivity,"
             "and deliver better employee experiences."
             "Your job is to collect and verify public information about the company {lead}, "
             "including its website, social media presence, and historical interactions with Merlin."
             "With these information on hand, your team will be able to dig deep into the opportunities"
             "working with that lead.",
    allow_delegation=False,
    verbose=True

)

lead_scoring_agent = Agent(
    role = "Lead Scoring",
    goal = "Score a lead analyzing some criteria",
    backstory = "You work at a company called goMerlin,a Brazilian technology company specializing in workforce management and optimization solutions for businesses. "
             "Their primary focus is on leveraging cutting-edge technology to enhance operational efficiency and streamline employee scheduling processes."
             "With a comprehensive suite of tools, goMerlin offers tailored solutions to meet the dynamic needs of companies in various industries."
             "Their platform is designed for businesses seeking to improve workforce management, enhance productivity,"
             "and deliver better employee experiences." 
             "goMerlin is willing to find new leads."
            "Your job is to give a score to a lead based on some criteria."
            "The criteria are:\n"
            "-The lead's size.\n"
            "-The lead's industry.\n"
            "-Preview interactions between Merlin and the lead.\n"
            "-The lead's social media presence and engagement.\n"
            "Please, describe with details the method you decide to use to determine the score.",
    allow_delegation = False,
    verbose = True

)

engagement_strategy_agent = Agent(
    role = "Engagement creator",
    goal = "Create good ways to engage with {lead}.",
    backstory = (
            "You work at a company called goMerlin,a Brazilian technology company specializing in workforce management and optimization solutions for businesses. "
             "Their primary focus is on leveraging cutting-edge technology to enhance operational efficiency and streamline employee scheduling processes."
             "With a comprehensive suite of tools, goMerlin offers tailored solutions to meet the dynamic needs of companies in various industries."
             "Their platform is designed for businesses seeking to improve workforce management, enhance productivity,"
             "and deliver better employee experiences." 
             "goMerlin is willing to find new leads."
             "Your job is to create a personalized approach for {lead},"
            "define the best communication channel, and propose ideal timing for contact."
    ),
    allow_delegation = False,
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
        linkedin_url = linkedin_lookup(lead)
        if not linkedin_url:
            return f"LinkedIn profile for {lead} not found."

        # Step 2: Scrape the LinkedIn URL
        scraped_content = scrape_tool.run(website_url=linkedin_url)
        return f"Scraped content from LinkedIn profile of {lead}:\n{scraped_content}"
    except Exception as e:
        return f"Error during LinkedIn scraping for {lead}: {str(e)}"

@tool
def twitter_scrape_tool(lead: str) -> str:
    """
    Finds the LinkedIn URL for the specified company and scrapes its content.
    Args:
        lead (str): The name of the company to search for.
    Returns:
        str: The scraped content from the company's LinkedIn profile.
    """
    try:
        # Step 1: Find the LinkedIn URL using the lookup function
        twitter_url = twitter_lookup(lead)
        if not twitter_url:
            return f"Twitter profile for {lead} not found."

        # Step 2: Scrape the LinkedIn URL
        scraped_content = scrape_tool.run(website_url=twitter_url)
        return f"Scraped content from Twitter profile of {lead}:\n{scraped_content}"
    except Exception as e:
        return f"Error during Twitter scraping for {lead}: {str(e)}"

research_task = Task(
    name="Public Information Research Task",
    description=(
        "Conduct a comprehensive public information research task on {lead}. This includes:\n\n"
        "- Collecting general public information about {lead}:\n"
        "- Review {lead}'s official website for details like mission, products/services, leadership, and history.\n"
        "- Identify {lead}'s size. Only for the {lead}'s size, trust the information from internet search results, not from Linkedin.\n"
        "- Identify the {lead}'s industry.\n\n"
        "- Verifying {lead}'s social media presence:\n"
        "- Analyze activity on LinkedIn. Look for the {lead}'s profile on Linkedin and scrape it. Collect information about its most active period.\n"
        "- Analyze which social media {lead} is active\n\n"
        "- Search for preview interactions between goMerlin and {lead}:\n"
        "- Searching for goMerlin and the lead's name ({lead}) side by side and scrape the most interesting links to get information about this previous contact.\n\n"
        "The agent will leverage these tools to deliver a structured and detailed report about {lead}, "
        "enabling stakeholders to make informed decisions based on accurate and comprehensive research."
    ),
expected_output=(
            "A structured report containing the following details:\n"
            "- General public information about {lead}.\n"
            "- Analysis of {lead}'s Linkedin and its engagement at this media."
            "- {lead}'s engagement at other social media.\n"
            "- History of interactions between Merlin and {lead}."
        ),
    tools=[search_tool, scrape_tool, linkedin_scrape_tool, file_read_tool],
    agent=lead_research_agent
)
scoring_task = Task(
    name="Scoring a Lead Task",
    description = (
        "Using the insights gathered from the lead researching on {lead}, "
        "give a score from 0 to 10 to {lead} based on some criteria. Each criteria should be evaluated from 0 to 10."
        "After that, tou should give weights from 0 to 1 to each criteria, based on their importance. The sum of all weights must be 1."
        "Then, calculate the weighted average considering the individual score for each criteria and its weight. The result is the final score.\n"
        "The criteria are:\n"
        "-The lead's size.\n"
        "-The lead's industry.\n"
        "-Preview interactions between Merlin and the lead.\n"
        "-The lead's social media presence and engagement.\n"
        "Please, describe with details the method you decide to use to determine the score.\n"
        "If you want, you can search the internet on how to score a lead.\n"
        "The function of the score is to indicate if making contact with the lead is a good opportunity for your company, goMerlin.\n"
        "Make sure that the method to determine the score makes sense!."
    ),
    expected_output = (
                    "A score from 0 to 10 that evaluates the impact that working with {lead} would bring to Merlin.\n"
                    "Also, a description about the method used to determine the score."
    ),
    tools=[search_tool, scrape_tool],
    agent=lead_scoring_agent
)

engagement_task = Task(
    name = "Make a strategy for a good engagement method",
    description = (
        "Using the insights gathered from the lead researching on {lead}, "
        "Your job is to create a personalized approach for {lead}. You must "
        "define the best communication channel based on the {lead} presence on social media. If the idea is the contact"
        "to be more formal, think about using e-mail. Also, propose ideal timing for contact."
    ),
    expected_output = (
        "You must give:\n"
        "-The best communication channel.\n"
        "-A suggestion of a personalized approach according to the communication channel chosen. For example, if the channel is e-mail, write and e-mail for them to start a collaboration. If it is instagram, write a direct message, etc.\n"
        "-The ideal timing for contact.\n"
        "Please, ensure to explain your choices with details. "
    ),
    tools=[search_tool, scrape_tool],
    agent=engagement_strategy_agent
)

crew = Crew(
    agents = [lead_research_agent, lead_scoring_agent, engagement_strategy_agent],
    tasks = [research_task, scoring_task, engagement_task],
    verbose=2,
    memory=True
)

inputs = {
    "lead": "iFood"
}

result = crew.kickoff(inputs=inputs)



