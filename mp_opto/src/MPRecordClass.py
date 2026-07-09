from src.MPOptoClass import *
class MPRecordClass(MPOptoClass):
    def __init__(self, session_path):
        try:
            CFA, RFA, analogin, isclimbing = load_mprecord(session_path)
        except Exception as e:
            print(e)
            return
        self.name = session_path.split('/')[-1]

        self.CFA = CFA
        self.RFA = RFA

        self.num_CFA = len(CFA['train'])
        self.num_RFA = len(RFA['train'])

        self.duration = analogin.shape[1] 
        
        self.climbing_bounds = climbing_bounds_from_logical(isclimbing)
        self.climbing_logical = isclimbing.astype(int)

        print(f'{self.name} successfulyl loaded')


