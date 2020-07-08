from flask import Blueprint, request, abort, jsonify
import json
from model import ActorCritic

app = Flask(__name__)

replay_buffer = []
max_buffer_size = 100
config = {}
actor_critic = ActorCritic(config)
is_new_policy_available = True

def serialize_model_parameter(m):
    return json.dumps(m.state_dict())

@app.route('/experience', methods=['POST'])
def post_experience():
    is_new_policy_available = False
    payload = request.json
    state = payload.get('state')
    next_state = payload.get('next_state')
    reward = payload.get('reward')
    action_prob_explore = payload.get('action_prob_explore')
    replay_buffer.append({
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action_prob_explore': action_prob_explore
    })

    if len(replay_buffer) > max_buffer_size:
        actor_critic.train(replay_buffer)
        replay_buffer = []

    return 'success'

@app.route('/policy/check', methods=['GET'])
def check_policy_update():
    return is_new_policy_available

@app.route('/policy/latest', methods=['GET'])
def get_latest_policy():
    return serialize_model_parameter(actor_critic)

if __name__ == "__main__":
    app.run(debug=True)
