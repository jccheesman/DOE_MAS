''' 
@jccheesman@alaska.edu
Run.py

This file runs and executes all notebooks at once. 
Each seperate code block is defined as a class
'''

import regionalization
import broad_overview_agent_discussion
import tsp_model
import pipeline

if __name__ == "__main__":
    regionalization.main()
    broad_overview_agent_discussion.main()  
    tsp_model.main()  
    pipeline.save_json()