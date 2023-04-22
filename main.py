from flask import Flask, jsonify, request
import replicate
import re
import os

rep_client = replicate.Client(api_token= os.environ['replicate_api'])
app = Flask(__name__)

@app.route('/predict')
def predict():
  
    user_text = request.args.get('input', '')
    instruction = request.args.get('instruction', '')
    model = request.args.get('model', '')
    
    prompt = 'instruction: ' + instruction + '\ninput: ' + user_text + '\noutput:'

    response = rep_client.run(model, input={"prompt": prompt, 
                                            "max_length": 100})

    # Output is returned as a stream
    output = []
  
    for text in response:
        output.append(text)
  
    # split the response into sentences
    sentences = re.split("(?<=[.!?]) +", ''.join(output))

    # remove any partial sentences that do not end with a punctuation mark
    sentences = [s for s in sentences if re.search("[.!?]$", s)]

    # join the remaining sentences back into a string
    final_response = ' '.join(sentences)
  
    return jsonify({'response': final_response})

# Run the app
if __name__ == '__main__':
    app.run(debug=True,
           host = '0.0.0.0')
