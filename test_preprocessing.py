from utils.preprocessing import Sentinel2Preprocessor

safe_path = r"C:\Users\JK\Downloads\S2B_MSIL2A_20211028T051929_N0500_R062_T44RKT_20230107T041610.SAFE"

processor = Sentinel2Preprocessor(safe_path)

processor.run_pipeline()
