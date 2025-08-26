from flask import Flask, request, jsonify
from flask_cors import CORS
import CoolProp.CoolProp as CP
import pickle

# ======================
# Load Pre-trained Model
# ======================
with open("compressor_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Define output labels
OUTPUT_LABELS = [
    "Volume Flow Rate (kg/hr)",
    "Compressor Shaft Power (kW)",
    "Torque (N-m)",
    "Adiabatic Efficiency (%)",
    "Volumetric Efficiency (%)",
    "Discharge Temperature (deg C)",
    "Volumetric Efficiency @3000 RPM (%)",
    "Adiabatic Efficiency @3000 RPM (%)",
    "Refrigeration Capacity @3000 RPM (kW)",
    "Compressor Power @3000 RPM  (kW)",
    "COP @3000 RPM"
]

# ======================
# Flask App
# ======================
app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        # ---------------------
        # Read Input Data
        # ---------------------
        data = request.json
        model_input = data.get("model")
        refrigerant = data.get("refrigerant")
        speed = float(data.get("speed"))
        superheat = float(data.get("superheat", 0))

        # Evaporator inputs (temp or pressure)
        evap_temp = data.get("evap_temp")  # °C
        suction_pressure = data.get("suction_pressure")  # bar abs

        # Condenser inputs (temp or pressure)
        cond_temp = data.get("cond_temp")  # °C
        discharge_pressure = data.get("discharge_pressure")  # bar abs

        # ---------------------
        # Calculate Evaporator Side Values
        # ---------------------
        if evap_temp is not None:
            Te_K = float(evap_temp) + 273.15
            Psuction = CP.PropsSI("P", "T", Te_K, "Q", 1, refrigerant)
        elif suction_pressure is not None:
            Psuction = float(suction_pressure) * 1e5
            Te_K = CP.PropsSI("T", "P", Psuction, "Q", 1, refrigerant)
        else:
            return jsonify({"error": "Either evap_temp or suction_pressure must be provided"})

        # ---------------------
        # Calculate Condenser Side Values
        # ---------------------
        if cond_temp is not None:
            Tc_K = float(cond_temp) + 273.15
            Pdischarge = CP.PropsSI("P", "T", Tc_K, "Q", 0, refrigerant)
        elif discharge_pressure is not None:
            Pdischarge = float(discharge_pressure) * 1e5
            Tc_K = CP.PropsSI("T", "P", Pdischarge, "Q", 0, refrigerant)
        else:
            return jsonify({"error": "Either cond_temp or discharge_pressure must be provided"})

        # ---------------------
        # Thermodynamic Calculations
        # ---------------------
        if superheat == 0:
            h1 = CP.PropsSI("H", "P", Psuction, "Q", 1, refrigerant)
            s1 = CP.PropsSI("S", "P", Psuction, "Q", 1, refrigerant)
            Tsuction_K = CP.PropsSI("T", "P", Psuction, "Q", 1, refrigerant)
        else:
            Tsuction_K = Te_K + superheat
            h1 = CP.PropsSI("H", "T", Tsuction_K, "P", Psuction, refrigerant)
            s1 = CP.PropsSI("S", "T", Tsuction_K, "P", Psuction, refrigerant)

        h2 = CP.PropsSI("H", "P", Pdischarge, "S", s1, refrigerant)
        h3 = CP.PropsSI("H", "P", Pdischarge, "Q", 0, refrigerant)

        # ---------------------
        # Prepare Model Input
        # ---------------------
        model_encoded = le.transform([model_input.strip().upper()])[0]
        input_features = [
            model_encoded,
            Te_K - 273.15,
            Psuction / 1e5,
            Tc_K - 273.15,
            Pdischarge / 1e5,
            speed
        ]
        input_scaled = scaler.transform([input_features])

        # ---------------------
        # ML Prediction
        # ---------------------
        predicted_output = model.predict(input_scaled)[0]
        predicted_results = dict(zip(OUTPUT_LABELS, predicted_output))

        # ---------------------
        # Performance Calculations
        # ---------------------
        flow_rate_kg_hr = predicted_results["Volume Flow Rate (kg/hr)"]
        flow_rate = flow_rate_kg_hr / 3600
        shaft_power = predicted_results["Compressor Shaft Power (kW)"]

        refrigeration_effect = (flow_rate * (h1 - h3) / 1000) if h1 and h3 else 0
        isentropic_work = (flow_rate * (h2 - h1) / 1000) if h1 and h2 else 0
        cop = refrigeration_effect / shaft_power if shaft_power else float('inf')

        # ---------------------
        # Final JSON Response
        # ---------------------
        result = {
            "Compressor Model": model_input,
            "Suction Pressure (bar abs)": round(Psuction / 1e5, 2),
            "Suction Temperature (°C)": round(Tsuction_K - 273.15, 2),
            "Discharge Pressure (bar abs)": round(Pdischarge / 1e5, 2),
            "Discharge Temperature (°C)": round(predicted_results["Discharge Temperature (deg C)"], 2),
            "Speed (RPM)": round(speed, 2),
            "Volume Flow Rate (kg/hr)": round(flow_rate_kg_hr, 2),
            "Compressor Shaft Power (kW)": round(shaft_power, 2),
            "Torque (N-m)": round(predicted_results["Torque (N-m)"], 2),
            "Adiabatic Efficiency (%)": round(predicted_results["Adiabatic Efficiency (%)"], 2),
            "Volumetric Efficiency (%)": round(predicted_results["Volumetric Efficiency (%)"], 2),
            "Volumetric Efficiency @3000 RPM (%)": round(predicted_results["Volumetric Efficiency @3000 RPM (%)"], 2),
            "Adiabatic Efficiency @3000 RPM (%)": round(predicted_results["Adiabatic Efficiency @3000 RPM (%)"], 2),
            "Refrigeration Capacity @3000 RPM (kW)": round(predicted_results["Refrigeration Capacity @3000 RPM (kW)"], 2),
            "Compressor Power @3000 RPM  (kW)": round(predicted_results["Compressor Power @3000 RPM  (kW)"], 2),
            "COP @3000 RPM": round(predicted_results["COP @3000 RPM"], 2),
            "Isentropic Work (kW)": round(isentropic_work, 2),
            "Refrigeration Effect (kW)": round(refrigeration_effect, 2),
            "Coefficient of Performance (COP)": round(cop, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ======================
# Run Flask App
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)