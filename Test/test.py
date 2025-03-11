import os
import logging
import asyncio
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import GitLoader
from langchain.prompts import PromptTemplate

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key="sk-proj-")

# Extract file structure
def extract_file_structure(path):
    def structure_helper(path):
        if os.path.basename(path) == ".git":
            return None
        structure = {}
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                structure[item] = structure_helper(item_path)
            else:
                structure[item] = None
        return structure
    return structure_helper(path)

# Load repository asynchronously
async def load_repository(repo_url, local_path, branch='main'):
    try:
        loader = GitLoader(clone_url=repo_url, repo_path=local_path, branch=branch)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} files from {repo_url}.")
        return documents
    except Exception as e:
        logger.error(f"Error loading repository: {e}")
        return []

# Generate summary for each file asynchronously
async def generate_file_summary(file_content, file_name):
    try:
        prompt = PromptTemplate.from_template("Summarize the given code file content in 100 words max: {content}")
        summary = (prompt | llm).invoke({"content": file_content})
        logger.info(f"Generated summary for {file_name}")
        return summary.content
    except Exception as e:
        logger.error(f"Error generating file summary for {file_name}: {e}")
        return ""

# Generate final project summary
def generate_project_summary(directory_summaries):
    try:
        prompt = PromptTemplate.from_template("Generate a comprehensive final project summary based on the following file summaries: {directories}")
        final_summary = (prompt | llm).invoke({"directories": '\n'.join(directory_summaries)})
        logger.info("Final project summary generated.")
        return final_summary.content
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        return ""

# Save final documentation
def save_final_documentation(summary, project_structure):
    try:
        with open("final_documentation.txt", "w", encoding="utf-8") as f:
            f.write("Project Structure:\n")
            f.write(str(project_structure) + "\n\n")
            f.write("Final Project Summary:\n")
            f.write(summary)
        logger.info("Final documentation saved.")
    except Exception as e:
        logger.error(f"Error saving final documentation: {e}")

# Hierarchical documentation agent
async def hierarchical_documentation_agent(repo_url, local_path, branch='main'):
    documents = await load_repository(repo_url, local_path, branch)
    file_summaries = []
    for doc in documents:
        summary = await generate_file_summary(doc.page_content, doc.metadata.get("source", "unknown"))
        if summary:
            file_summaries.append(summary)
    final_summary = generate_project_summary(file_summaries)
    project_structure = extract_file_structure(local_path)
    save_final_documentation(final_summary, project_structure)

# Entry point
repo_url = "https://github.com/langchain-ai/web-explorer"
local_path = "./example_data/test_repo"
asyncio.run(hierarchical_documentation_agent(repo_url, local_path))
