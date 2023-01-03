def get_eval_traj(cached_state_path, plan_step=3):
    """Test set"""
    if 'cutrearrangespread' in cached_state_path:
        if plan_step == 3:
            init_v = [600, 601, 602, 603, 604]
            target_v = [600, 601, 602, 603, 604]
        else:
            assert plan_step == 6
            init_v = [610, 611, 612, 613, 614]
            target_v = [610, 611, 612, 613, 614]
        return init_v, target_v
    if '1215_cutrearrange' in cached_state_path:
        # The first 4 are three-stage and the last one is two-stage
        init_v = [0, 3, 6, 9, 6]
        target_v = [2, 5, 8, 11, 7]
        return init_v, target_v
    elif '1215_gathermove' in cached_state_path:
        init_v = [0, 2, 4, 8, 16]
        target_v = [102, 103, 102, 103, 104]
        return init_v, target_v
    elif '0923_LiftSpread' in cached_state_path or '1215_liftspread' in cached_state_path or '1230_liftspread' in cached_state_path or '0202_liftspread' in cached_state_path:
        init_v_1 = [104, 104, 106, 106, 107]
        target_v_1 = [109, 112, 143, 166, 185]
        return init_v_1, target_v_1
    elif '0926_GatherMove' in cached_state_path or '0202_gathermove' in cached_state_path or '0307_gathermove' in cached_state_path:
        init_v = [0, 2, 4, 8, 16]
        target_v = [102, 103, 102, 103, 104]
        return init_v, target_v
    elif 'CutRearrange' in cached_state_path:
        init_v = [24, 27, 33, 42, 51]
        target_v = [25, 28, 34, 43, 52]
        return init_v, target_v
    elif '1116_Lift' in cached_state_path:
        init_v = [24, 27, 33, 42, 51]
        target_v = [24, 37, 13, 42, 71]
        return init_v, target_v
    elif '1116_Spread' in cached_state_path  or '1118_Spread_correct' in cached_state_path:
        init_v = [24, 27, 33, 42, 51]
        target_v = [24, 37, 13, 42, 71]
        return init_v, target_v
    else:
        raise NotImplementedError

def get_eval_skill_trajs(cached_state_path, tid):
    if '1215_cutrearrange' in cached_state_path:
        if tid == 0:
            init_v = [0, 3, 6, 9, 12]
            target_v = [0, 3, 6, 9, 12]
        else:
            init_v = [1, 2, 4, 8, 10]
            target_v = [1, 2, 4, 8, 10]
        return init_v, target_v
    elif '0202_liftspread' in cached_state_path:
        if tid == 0:
            init_v = [27, 35, 38, 46, 48]
            target_v = [109, 112, 143, 166, 185]
        else:
            init_v = [104, 104, 106, 106, 107]
            target_v = [27, 35, 38, 46, 48]
        return init_v, target_v
    elif '0202_gathermove' in cached_state_path:
        if tid == 0:
            init_v = [0, 2, 4, 8, 16]
            target_v = [1, 3, 5, 9, 17]
        else:
            init_v = [1, 3, 5, 9, 17]
            target_v = [102, 103, 102, 103, 104]
        return init_v, target_v
    elif 'cutrearrangespread' in cached_state_path:
        if tid == 0:
            init_v = [0, 1, 2, 3, 4]
            target_v = [0, 1, 2, 3, 4]
        elif tid == 1:
            init_v = [200, 201, 202, 203, 204]
            target_v = [200, 201, 202, 203, 204]
        else:
            init_v = [400, 401, 402, 403, 404]
            target_v = [400, 401, 402, 403, 404]
        return init_v, target_v
    else:
        raise NotImplementedError