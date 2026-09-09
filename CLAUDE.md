Please see @data/TASK.pdf, we will need to build a frontend web-app that output following deliverable:             
  1. Global Recommendation (All users have the same)                                                                 
  2. User Recommendation (User dependent)                                                                            
                                                                                                                     
  Help me build the abstraction so we can switch and iterate the recommendation system algorithm easily.             
                                                                                                                     
  I think in the "Functional Requirements" already has good function abstraction to adopt: recommend_popular and     
  recommend_for_user. Since we will build frontend web-app, we will pick Web API interface for the backend system    
  where the recommendation system component lives.                                                                   
                                                                                                                     
  Read section "2.3 Optional/Nice-to-Have", we need to make the system interpretable. We will approach this two way: 
  1. Algorithm behavior first-principled (Will think about this later, we will interpret how the algorithm work)     
  2. Calling a generative LLM to explain/interpret the prediction made by our recommendation system (This defined in 
  @data/TASK_2.pdf)                                                                                                  
                                                                                                                     
  In order to comply with section "Non-Functional Requirements", we need /brainstorming and properly scope the       
  project structure. Let me control and nitpick the project and software architecture. You must surface and confirm  
  to me on this.                                                                                                     
                                                                                                                     
  First, please propose the software architecture starting from project structure enabling plug and play             
  recommendation system algorithm.