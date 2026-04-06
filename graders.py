from env import TrafficEnv

def collect_history(task_id, action_fn):
    env = TrafficEnv()
    state = env.reset(task_id)
    
    history = []
    
    for _ in range(100):
        action = action_fn(state)
        state, reward, done, info = env.step(action)  # ✅ 4 values
        history.append(state)
        
        if done:  # ✅ episode khatam ho gaya
            break
    
    return history

def calculate_score(history):
    north_avg = sum(step["north_wait"] for step in history) / len(history)
    south_avg = sum(step["south_wait"] for step in history) / len(history)
    east_avg = sum(step["east_wait"] for step in history) / len(history)
    west_avg = sum(step["west_wait"] for step in history) / len(history)
    
    overall_avg = (north_avg + south_avg + east_avg + west_avg) / 4
    
    score = 1.0 - (overall_avg - 10) / 170
    score = max(0.0, min(1.0, score))
    
    return score

def grade_all(action_fn):
    for task_id in [1, 2, 3]:  # ✅ integers now
        history = collect_history(task_id, action_fn)
        score = calculate_score(history)
        print(f"task_{task_id}: {score:.2f}")
