"""
Student Focus Behavior Dataset Generator
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(42)
N = 180

def generate_dataset():
    session_duration_min = np.random.randint(10, 120, N)
    break_frequency = np.random.randint(0, 6, N)
    phone_pickups = np.random.randint(0, 20, N)
    tab_switches = np.random.randint(0, 30, N)
    task_completion_rate = np.round(np.random.beta(2, 1.5, N), 2)
    self_rated_focus = np.random.randint(1, 11, N)
    background_noise_level = np.random.choice(["silent", "low", "medium", "high"], N, p=[0.3, 0.3, 0.25, 0.15])
    time_of_day = np.random.choice(["morning", "afternoon", "evening", "night"], N, p=[0.3, 0.25, 0.25, 0.2])
    days_since_last_break = np.random.randint(0, 5, N)
    caffeine_intake_cups = np.random.choice([0, 1, 2, 3, 4], N, p=[0.2, 0.35, 0.3, 0.1, 0.05])
    prior_sleep_hours = np.round(np.random.normal(6.5, 1.5, N).clip(3, 10), 1)
    streak_days = np.random.randint(0, 30, N)

    # More balanced label generation
    # High focus: good sleep + high self-rated + low distractions + high completion
    high_focus_prob = (
        (task_completion_rate > 0.65).astype(float) * 0.35
        + (self_rated_focus >= 7).astype(float) * 0.30
        + (prior_sleep_hours >= 6.5).astype(float) * 0.15
        + (phone_pickups < 5).astype(float) * 0.10
        + (tab_switches < 8).astype(float) * 0.10
    )
    high_focus_prob = np.clip(high_focus_prob + np.random.normal(0, 0.1, N), 0, 1)
    label = (high_focus_prob >= 0.6).astype(int)

    df = pd.DataFrame({
        "session_duration_min": session_duration_min,
        "break_frequency": break_frequency,
        "phone_pickups": phone_pickups,
        "tab_switches": tab_switches,
        "task_completion_rate": task_completion_rate,
        "self_rated_focus": self_rated_focus,
        "background_noise_level": background_noise_level,
        "time_of_day": time_of_day,
        "days_since_last_break": days_since_last_break,
        "caffeine_intake_cups": caffeine_intake_cups,
        "prior_sleep_hours": prior_sleep_hours,
        "streak_days": streak_days,
        "high_focus": label
    })

    df = shuffle(df, random_state=42).reset_index(drop=True)
    df.to_csv("data/student_focus_behavior.csv", index=False)
    print(f"Dataset saved: {len(df)} rows | High-focus: {df['high_focus'].mean():.1%} | Low-focus: {(1-df['high_focus']).mean():.1%}")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df["high_focus"].value_counts())
