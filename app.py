from flask import Flask, request, jsonify
from flask_cors import CORS
import CoolProp.CoolProp as CP
import pickle
import io
import base64
import matplotlib.pyplot as plt
import numpy as np

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

@app.route('/plot_economizer', methods=['POST'])
def plot_economizer():
    try:
        data = request.json

        # Extract required values from /process output
        refrigerant = data.get("Refrigerant")
        h1 = data.get("h1")
        h2 = data.get("h2")
        h3 = data.get("h3")
        Ps_bar = data.get("Suction Pressure (bar abs)")
        Pd_bar = data.get("Discharge Pressure (bar abs)")

        if None in [refrigerant, h1, h2, h3, Ps_bar, Pd_bar]:
            return jsonify({"error": "Missing required fields. Please pass full /process output."})

        # Convert bar to Pa
        P1 = Ps_bar * 1e5
        P2 = Pd_bar * 1e5
        P3 = P2
        P4 = P1
        h4 = h3

        # Economizer state point (assume mid-pressure)
        Pecon = np.sqrt(P1 * P2)
        h_econ_inj = CP.PropsSI("H", "P", Pecon, "Q", 1, refrigerant)

        # Cycle data including economizer injection
        h_cycle = [h1, h2, h3, h4, h1]
        p_cycle = [P1, P2, P3, P4, P1]

        # Saturation curve
        p_vals = np.logspace(np.log10(P1 * 0.5), np.log10(P2 * 1.5), 300)
        hL = [CP.PropsSI("H", "P", p, "Q", 0, refrigerant) for p in p_vals]
        hV = [CP.PropsSI("H", "P", p, "Q", 1, refrigerant) for p in p_vals]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hL, np.array(p_vals)/1e5, 'b-', label='Saturated Liquid')
        ax.plot(hV, np.array(p_vals)/1e5, 'r-', label='Saturated Vapor')
        ax.plot(h_cycle, np.array(p_cycle)/1e5, 'ko-', linewidth=2, label='Refrigeration Cycle')

        # Economizer point
        ax.plot(h_econ_inj, Pecon/1e5, 'gs', markersize=8, label='Economizer Injection')

        # Labels
        labels = ["1", "2", "3", "4"]
        for i, (h, p) in enumerate(zip(h_cycle[:-1], p_cycle[:-1])):
            ax.annotate(f"{labels[i]}\n{h/1000:.1f} kJ/kg", (h, p/1e5), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.6))

        ax.annotate(f"Econ Inj\n{h_econ_inj/1000:.1f} kJ/kg", 
                    (h_econ_inj, Pecon/1e5), xytext=(-40, -20),
                    textcoords='offset points', ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.6))

        ax.set_xlabel("Enthalpy (J/kg)")
        ax.set_ylabel("Pressure (bar abs)")
        ax.set_yscale("log")
        ax.set_title(f"P-h Diagram with Economizer for {refrigerant}")
        ax.legend()
        ax.grid(True, which="both", ls="--", lw=0.5)

        # Encode image
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300)
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        # Economizer result
        result = {
            "Economizer Pressure (bar abs)": round(Pecon/1e5, 2),
            "Economizer Enthalpy (kJ/kg)": round(h_econ_inj/1000, 2),
            "ph_diagram_economizer": img_base64
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.json
        model_input = data.get("model")
        refrigerant = data.get("refrigerant")
        speed = float(data.get("speed"))
        superheat = float(data.get("superheat", 0))

        # Evaporator inputs
        evap_temp = data.get("evap_temp")
        suction_pressure = data.get("suction_pressure")

        # Condenser inputs
        cond_temp = data.get("cond_temp")
        discharge_pressure = data.get("discharge_pressure")

        # Evaporator calculations
        if evap_temp is not None:
            Te_K = float(evap_temp) + 273.15
            Psuction = CP.PropsSI("P", "T", Te_K, "Q", 1, refrigerant)
        elif suction_pressure is not None:
            Psuction = float(suction_pressure) * 1e5
            Te_K = CP.PropsSI("T", "P", Psuction, "Q", 1, refrigerant)
        else:
            return jsonify({"error": "Either evap_temp or suction_pressure must be provided"})

        # Condenser calculations
        if cond_temp is not None:
            Tc_K = float(cond_temp) + 273.15
            Pdischarge = CP.PropsSI("P", "T", Tc_K, "Q", 0, refrigerant)
        elif discharge_pressure is not None:
            Pdischarge = float(discharge_pressure) * 1e5
            Tc_K = CP.PropsSI("T", "P", Pdischarge, "Q", 0, refrigerant)
        else:
            return jsonify({"error": "Either cond_temp or discharge_pressure must be provided"})

        # Thermodynamic calculations
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

        # Prepare model input
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

        # ML Prediction
        predicted_output = model.predict(input_scaled)[0]
        predicted_results = dict(zip(OUTPUT_LABELS, predicted_output))

        # Performance Calculations
        flow_rate_kg_hr = predicted_results["Volume Flow Rate (kg/hr)"]
        flow_rate = flow_rate_kg_hr / 3600
        shaft_power = predicted_results["Compressor Shaft Power (kW)"]

        refrigeration_effect = (flow_rate * (h1 - h3) / 1000) if h1 and h3 else 0
        isentropic_work = (flow_rate * (h2 - h1) / 1000) if h1 and h2 else 0
        cop = refrigeration_effect / shaft_power if shaft_power else float('inf')

        # Final JSON Response
        result = {
            "Compressor Model": model_input,
            "Refrigerant": refrigerant,
            "Suction Pressure (bar abs)": round(Psuction / 1e5, 2),
            "Discharge Pressure (bar abs)": round(Pdischarge / 1e5, 2),
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "evap_temp": float(evap_temp) if evap_temp is not None else None,
            "cond_temp": float(cond_temp) if cond_temp is not None else None
        }

        result.update(predicted_results)
        result["Isentropic Work (kW)"] = round(isentropic_work, 2)
        result["Refrigeration Effect (kW)"] = round(refrigeration_effect, 2)
        result["Coefficient of Performance (COP)"] = round(cop, 2)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/plot', methods=['POST'])
def plot_ph_diagram():
    try:
        data = request.json

        # Extract required values from /process output
        refrigerant = data.get("Refrigerant")
        h1 = data.get("h1")
        h2 = data.get("h2")
        h3 = data.get("h3")
        Ps_bar = data.get("Suction Pressure (bar abs)")
        Pd_bar = data.get("Discharge Pressure (bar abs)")

        if None in [refrigerant, h1, h2, h3, Ps_bar, Pd_bar]:
            return jsonify({"error": "Missing required fields. Please pass full /process output."})

        # Convert bar to Pa
        P1 = Ps_bar * 1e5
        P2 = Pd_bar * 1e5
        P3 = P2
        P4 = P1
        h4 = h3

        # Cycle data
        h_cycle = [h1, h2, h3, h4, h1]
        p_cycle = [P1, P2, P3, P4, P1]

        # Saturation curve
        p_vals = np.logspace(np.log10(P1 * 0.5), np.log10(P2 * 1.5), 300)
        hL = [CP.PropsSI("H", "P", p, "Q", 0, refrigerant) for p in p_vals]
        hV = [CP.PropsSI("H", "P", p, "Q", 1, refrigerant) for p in p_vals]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hL, np.array(p_vals)/1e5, 'b-', label='Saturated Liquid')
        ax.plot(hV, np.array(p_vals)/1e5, 'r-', label='Saturated Vapor')
        ax.plot(h_cycle, np.array(p_cycle)/1e5, 'ko-', linewidth=2, label='Refrigeration Cycle')

        labels = ["1", "2", "3", "4"]
        for i, (h, p) in enumerate(zip(h_cycle[:-1], p_cycle[:-1])):
            ax.annotate(f"{labels[i]}\n{h/1000:.1f} kJ/kg", (h, p/1e5), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.6))

        ax.set_xlabel("Enthalpy (J/kg)")
        ax.set_ylabel("Pressure (bar abs)")
        ax.set_yscale("log")
        ax.set_title(f"P-h Diagram for {refrigerant}")
        ax.legend()
        ax.grid(True, which="both", ls="--", lw=0.5)

        # Encode image
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300)
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return jsonify({"ph_diagram": img_base64})

    except Exception as e:
        return jsonify({"error": str(e)})

# ======================
# Run Flask App
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
