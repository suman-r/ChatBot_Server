from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import ollama

app = Flask(__name__)
CORS(app, origins=['http://localhost:3005'])

MEMORY_FILE = 'chat_memory.json'
MODEL = 'llama3.2'
PAGE_SIZE = 20

# --- Utility Functions ---
def load_all_memories():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_all_memories(all_memories):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(all_memories, f, indent=2)

# --- API Endpoints ---
@app.route('/api/instances', methods=['GET'])
def get_instances():
    all_memories = load_all_memories()
    return jsonify(list(all_memories.keys()))

@app.route('/api/instances', methods=['POST'])
def create_instance():
    data = request.json
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()

    if not name:
        return jsonify({'error': 'Character name is required'}), 400

    all_memories = load_all_memories()
    if name in all_memories:
        return jsonify({'error': 'Character already exists'}), 400

    all_memories[name] = [{'role': 'system', 'content': description or 'You are a helpful assistant.'}]
    save_all_memories(all_memories)
    return jsonify({'message': 'Character created successfully.', 'ok': True}), 201

@app.route('/api/instances/<instance_name>', methods=['DELETE'])
def delete_instance(instance_name):
    all_memories = load_all_memories()
    if instance_name not in all_memories:
        return jsonify({'error': 'Character not found'}), 404

    del all_memories[instance_name]
    save_all_memories(all_memories)
    return jsonify({'message': f'Character "{instance_name}" deleted successfully.', 'ok': True}), 200

@app.route('/api/chat/<instance_name>', methods=['GET'])
def get_messages(instance_name):
    offset = int(request.args.get('offset', 0))
    all_memories = load_all_memories()
    messages = all_memories.get(instance_name, [])
    paginated = messages[::-1][offset:offset + PAGE_SIZE][::-1]  # Return most recent PAGE_SIZE messages
    return jsonify(paginated)

@app.route('/api/chat/<instance_name>', methods=['POST'])
def send_message(instance_name):
    data = request.json
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({'error': 'Message cannot be empty'}), 400

    all_memories = load_all_memories()
    if instance_name not in all_memories:
        return jsonify({'error': 'Instance not found'}), 404

    messages = all_memories[instance_name]
    messages.append({'role': 'user', 'content': user_input})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            'temperature': 0.9,
            'top_p': 0.95,
            'num_predict': 1024
        }
    )

    assistant_reply = response['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    all_memories[instance_name] = messages
    save_all_memories(all_memories)

    return jsonify({'reply': assistant_reply, 'ok': True}), 200

if __name__ == '__main__':
    app.run(debug=True)