from env import TrafficEnv

def collect_history(task_id, action_fn):
    env = TrafficEnv()
    state = env.reset(task_id)
    
    history = []
    
    for _ in range(100):  # 100 steps
        action = action_fn(state)  # bahar se action function aayega
        state, reward = env.step(action)
        history.append(state)
    
    return history





def calculate_score(history):
    
    # Step 1: Har lane ka average wait time
    north_avg = sum(step["north_wait"] for step in history) / len(history)
    south_avg = sum(step["south_wait"] for step in history) / len(history)
    east_avg = sum(step["east_wait"] for step in history) / len(history)
    west_avg = sum(step["west_wait"] for step in history) / len(history)
    
    # Step 2: Overall average
    overall_avg = (north_avg+south_avg+east_avg+west_avg)/4
    
    # Step 3: Score
    score = 1.0 - (overall_avg - 10) / (170)
    score = max(0.0, min(1.0, score))
    
    return score

def grade_all(action_fn):
    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        history = collect_history(task_id, action_fn)
        score = calculate_score(history)
        print(f"{task_id}: {score:.2f}")    