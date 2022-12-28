import time

def render_one_game(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action.item())
        env.render()
        time.sleep(.01)
