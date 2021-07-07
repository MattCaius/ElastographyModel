from flask import Flask, jsonify, request

app = Flask(__name__)

#Routes external JS request  to the predictor
@app.route('/flask/<filename>', methods=['GET'])
def index(filename):
    print(filename)
    #The filename is attached in the request. It gets routed to the predictfile function 
    import Predictor
    output=Predictor.predictfile(filename)
    #Test the output
    print(output)
    #Returns the json output
    return jsonify({'output': output})

if __name__ == "__main__":
    #Run it in port 5000
    app.run(port=5000, debug=True)