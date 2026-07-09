from src.MPOptoClass import *
class ReachingClass(MPOptoClass):
    def __init__(self, session_path, pre=200, post=799):
        CFA, RFA, reach_bounds = load_reaching(session_path)
        reach_bounds[:,0] = reach_bounds[:,0] - pre
        reach_bounds[:,1] = reach_bounds[:,0] + post
        
        self.name = session_path.split('/')[-1]
        self.CFA = CFA
        self.RFA = RFA
        self.num_CFA = len(CFA['train'])
        self.num_RFA = len(RFA['train'])
        self.reach_bounds = reach_bounds

