from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('models/rfc_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

def preprocess_data(data):
    def ubahHomeownership(row):
        if row == "RENT":
            return 0
        elif row == "OWN":
            return 1
        elif row == "MORTGAGE":
            return 2
        elif row == "OTHER":
            return 3
        elif row == "NONE":
            return 4
        elif row == "ANY":
            return 5

    def ubahVerifstatus(row):
        if row == "Verified":
            return 0
        elif row == "Source Verified":
            return 1
        elif row == "Not Verified":
            return 2

    def ubahListstatus(row):
        if row == "f":
            return 0
        elif row == "w":
            return 1

    purpose = {
        'credit_card': 0,
        'car': 1,
        'small_business': 2,
        'other': 3,
        'wedding': 4,
        'debt_consolidation': 5,
        'home_improvement': 6,
        'major_purchase': 7,
        'medical': 8,
        'moving': 9,
        'vacation': 10,
        'house': 11,
        'renewable_energy': 12,
        'educational': 13
    }

    addr_state = {
        'AZ': 0, 'GA': 1, 'IL': 2, 'CA': 3, 'OR': 4, 'NC': 5, 'TX': 6, 'VA': 7, 'MO': 8,
        'CT': 9, 'UT': 10, 'FL': 11, 'NY': 12, 'PA': 13, 'MN': 14, 'NJ': 15, 'KY': 16,
        'OH': 17, 'SC': 18, 'RI': 19, 'LA': 20, 'MA': 21, 'WA': 22, 'WI': 23, 'AL': 24,
        'CO': 25, 'KS': 26, 'NV': 27, 'AK': 28, 'MD': 29, 'WV': 30, 'VT': 31, 'MI': 32,
        'DC': 33, 'SD': 34, 'NH': 35, 'AR': 36, 'NM': 37, 'MT': 38, 'HI': 39, 'WY': 40,
        'OK': 41, 'DE': 42, 'MS': 43, 'TN': 44, 'IA': 45, 'NE': 46, 'ID': 47, 'IN': 48,
        'ME': 49
    }

    data['home_ownership'] = data['home_ownership'].apply(ubahHomeownership).astype(int)
    data['verification_status'] = data['verification_status'].apply(ubahVerifstatus).astype(int)
    data['purpose'] = data['purpose'].map(purpose).astype(int)
    data['addr_state'] = data['addr_state'].map(addr_state).astype(int)
    data['initial_list_status'] = data['initial_list_status'].apply(ubahListstatus).astype(int)

    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari request JSON
    data = request.get_json()
    
    # Debugging: Print data yang diterima
    print("Data yang diterima:", data)
    
    data_df = pd.DataFrame([data])

    # Preprocess data
    data_preprocessed = preprocess_data(data_df)
    
    # Pisahkan kolom numerik dan kolom kategori
    numerical_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 
                      'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'total_rec_late_fee', 
                      'recoveries', 'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 
                      'emp_length_int', 'term_int', 'mths_since_earliest_cr_line', 'mths_since_issue_d']
    categorical_cols = ['home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status']
    
    # Skala data numerik
    numerical_data = data_preprocessed[numerical_cols].astype(float)  # pastikan tipe data numerik
    scaled_data = pd.DataFrame(scaler.transform(numerical_data), columns=numerical_cols)
    
    # Gabungkan data yang di-encode secara manual dan data yang di-skala
    encoded_data = data_preprocessed[categorical_cols]
    final_data = pd.concat([encoded_data, scaled_data], axis=1)
    
    # Prediksi menggunakan model
    prediction = model.predict(final_data)
    
    # Ubah hasil prediksi menjadi tipe data dasar
    prediction = int(prediction[0])
    
    # Konversi hasil prediksi ke nasabah baik atau buruk
    result = "Good Loan" if prediction == 0 else "Bad Loan"

    # Kembalikan hasil prediksi dalam format JSON
    return jsonify({'prediction': result})

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)