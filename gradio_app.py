import gradio as gr
import torch
import numpy as np
import os
import pickle
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import glob
# 导入模型和配置
from model import MTS_LOF_revised
from config_files.SHHS1_Configs import Config
from frs_compute.frs import frs

class GradioInference:
    def __init__(self):
        self.configs = Config()
        self.configs.mutiple_models = True
        self.configs.channel_list = ['ECG', 'EEG', 'Resp']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transform_vectors = {}
        self.scalers = {}
        
        # Disease type mapping
        self.disease_names = {
            0: 'MI (Myocardial Infarction)',
            1: 'RBBB (Right Bundle Branch Block)', 
            2: 'CHF (Congestive Heart Failure)',
            3: 'Hypertension',
        }
        
        # 阳性平均得分数据（写死的对比数据）
        self.positive_average_scores = {
            0: {'ECG': 1.17, 'EEG': 1.19, 'Resp': 1.21},  # MI
            1: {'ECG': 1.5549, 'EEG': 1.2239, 'Resp': 1.2563},  # RBBB
            2: {'ECG': 1.2888, 'EEG': 1.2504, 'Resp': 1.2824},  # CHF
            3: {'ECG': 1.1108, 'EEG': 1.1508, 'Resp': 1.2001},  # Hypertension
        }
        
        self.load_models()
        print('begin')
        self.load_transform_vectors()
    
    def load_models(self):
        """Load pre-trained models"""
        print("Loading models...")
        data_type = 'SHHS1'
        seed = 2019
        
        for channel in self.configs.channel_list:
            print(f"Loading {channel} model...")
            self.configs.channel = channel
            
            # Set model parameters
            if channel == 'Resp':
                self.configs.kernel_size = 15
                self.configs.stride = 2
            else:
                self.configs.kernel_size = 25
                self.configs.stride = 6
            
            model = MTS_LOF_revised(self.configs)
            
            # Load pre-trained weights
            model_path = f'./checkpoints/{data_type}/ssl_{seed}_embed_dim_normalize{self.configs.embed_dim}{channel}30seconds.pth'
            if os.path.exists(model_path):
                pretrained_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(pretrained_dict, strict=False)
                print(f"Successfully loaded {channel} model from {model_path}")
            else:
                print(f"Warning: Model file not found: {model_path}")
                return False
            
            model.eval()
            model = model.to(self.device)
            self.models[channel] = model
            print(f"Successfully loaded {channel} model")
        
        return True
    
    def load_transform_vectors(self):
        """Load transform vectors and scalers"""
        print("Loading transform vectors and scalers...")
        save_dir = './projections'
        for active_label_set in range(4):
            for channel in self.configs.channel_list:
                # Find the first .pt file
                pt_files = sorted(glob.glob(os.path.join(save_dir, "*.pt")))
                if len(pt_files) == 0:
                    raise FileNotFoundError(f"No .pt file found in {save_dir}")
                results_path = pt_files[0]
                if os.path.exists(results_path):
                    print(results_path)
                    saved_results = torch.load(results_path, map_location=self.device,weights_only=False)
                    print(f"Successfully loaded {channel} transform vectors")
                    # Load transform vectors
                    if 'transform_vectors' in saved_results:
                        for key, vector in saved_results['transform_vectors'].items():
                            print('success')
                            print(key)
                            self.transform_vectors[f"{key}"] = vector.to(self.device)
                    
                    # Load scalers
                    for scaler_name in ['age_scaler', 'score_scaler', 'frs_score_scaler', 'bmi_scaler']:
                        if scaler_name in saved_results:
                            self.scalers[f"{channel}_{active_label_set}_{scaler_name}"] = saved_results[scaler_name]
                    
                    print(f"Loaded vectors for {channel} - {self.disease_names[active_label_set]}")
                else:
                    print(f"Warning: Transform vectors file not found: {results_path}")
    
    def preprocess_signal(self, signal_data, channel):
        """Preprocess signal data"""
        # Ensure signal is numpy array
        if isinstance(signal_data, list):
            signal_data = np.array(signal_data)
        
        # Convert to tensor and add batch and channel dimensions
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
        signal_tensor = signal_tensor.unsqueeze(1)
        
        return signal_tensor
    
    def extract_patient_info_from_pt(self, pt_data):
        """Extract patient information from .pt file"""
        patient_info = {}
        
        # Extract patient information fields
        info_fields = {
            'age': 'age',
            'sex': 'sex', 
            'bmi': 'bmi',
            'chol': 'chol',
            'hdl': 'hdl',
            'systbp': 'systbp',
            'parrptdiab': 'parrptdiab',
            'smoke': 'smokstat_s1',  # Note the field name here
            'medicine': 'htnmed1'    # Note the field name here
        }
        
        for key, pt_key in info_fields.items():
            if pt_key in pt_data:
                value = pt_data[pt_key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                patient_info[key] = value
        
        return patient_info
    
    def calculate_projection_scores(self, eeg_data, ecg_data, resp_data, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine):
        """Calculate projection scores for all diseases"""
        try:
            # Preprocess signals
            signals = {}
            if eeg_data is not None:
                signals['EEG'] = self.preprocess_signal(eeg_data, 'EEG')
            if ecg_data is not None:
                signals['ECG'] = self.preprocess_signal(ecg_data, 'ECG')
            if resp_data is not None:
                signals['Resp'] = self.preprocess_signal(resp_data, 'Resp')
            # Calculate FRS score (gender: 1=male, 2=female)
            frs_gender = 1 if sex == 1 else 2
            frs_score = frs(frs_gender, 10, age, chol, hdl, systbp, parrptdiab, smoke, medicine)
            print(frs_score,signals['EEG'].shape)
            results = {}
            # Calculate projection scores for each disease
            for active_label_set in range(4):
                disease_name = self.disease_names[active_label_set]
                results[disease_name] = {}
                for channel, signal in signals.items():
                    if channel in self.models:
                        model = self.models[channel].to(self.device)
                        signal = signal.to(self.device)
                        try:
                            # Get features
                            with torch.no_grad():
                                _, feat = model(signal)
                                print(f"Model {channel} output shape: {feat.shape}")
                                feat_norm = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
                            # Find transform vector
                            key = f"{channel}_{active_label_set}_0->1"
                            if key in self.transform_vectors:
                                transform_vector = self.transform_vectors[key]
                                print(f"Transform vector {key} shape: {transform_vector.shape}")
                                transform_vector = transform_vector / torch.norm(transform_vector, p=2)
                                # Calculate projection score
                                score = torch.matmul(feat_norm, transform_vector).cpu()
                                score = torch.exp(torch.topk(score, 3)[0].mean())
                                results[disease_name][channel] = float(score.item())
                            else:
                                print(f"Transform vector not found: {key}")
                                results[disease_name][channel] = "No transform vector available"
                        except Exception as e:
                            print(f"Error processing {channel} for {disease_name}: {str(e)}")
                            results[disease_name][channel] = f"Error: {str(e)}"
            
            # Add patient information
            results['Patient_Info'] = {
                'Age': age,
                'Sex': 'Male' if sex == 1 else 'Female',
                'BMI': bmi,
                'Cholesterol': chol,
                'HDL': hdl,
                'Systolic_BP': systbp,
                'Diabetes': 'Yes' if parrptdiab == 1 else 'No',
                'Smoking': 'Yes' if smoke == 1 else 'No',
                'Medicine': 'Yes' if medicine == 1 else 'No',
                'FRS_Score': frs_score
            }
            return results
            
        except Exception as e:
            return {"Error": str(e)}
    
    def format_results_with_comparison(self, results, detected_signals=None, extracted_patient_info=False, is_combined_mode=False):
        """Format results with comparison to average scores and warnings"""
        output_text = "=== Projection Score Analysis Results ===\n\n"
        
        # Add data source information
        if is_combined_mode:
            output_text += "ℹ️ Using the unified .pt file mode\n"
        if extracted_patient_info:
            output_text += "ℹ️ Patient information has been automatically extracted from the file\n"
        output_text += "\n"
        
        # Add signal detection information
        if detected_signals:
            output_text += f"📡 Detected signals: {', '.join(detected_signals)}\n\n"
        
        # Add patient information
        if 'Patient_Info' in results:
            output_text += "Patient Information:\n"
            for key, value in results['Patient_Info'].items():
                output_text += f"  {key}: {value}\n"
            output_text += "\n"
        
        # Add disease projection scores with comparison
        output_text += "Projection Scores for Each Disease:\n"
        high_risk_warnings = []
        
        for disease, scores in results.items():
            if disease != 'Patient_Info' and disease != 'Error':
                output_text += f"\n{disease}:\n"
                if isinstance(scores, dict):
                    disease_id = None
                    # Find disease ID
                    for d_id, d_name in self.disease_names.items():
                        if d_name == disease:
                            disease_id = d_id
                            break
                    
                    for channel, score in scores.items():
                        if isinstance(score, (int, float)):
                            # Get average score for comparison
                            if disease_id is not None and disease_id in self.positive_average_scores:
                                avg_score = self.positive_average_scores[disease_id].get(channel, 0)
                                
                                # Compare with average and add warning if higher
                                if score > avg_score:
                                    risk_ratio = score / avg_score
                                    warning_level = "🔴 HIGH RISK" if risk_ratio > 1.5 else "🟡 ELEVATED"
                                    output_text += f"  {channel}: {score:.6f} {warning_level} (avg: {avg_score:.4f}, ratio: {risk_ratio:.2f}x)\n"
                                    high_risk_warnings.append(f"{disease} - {channel}: {risk_ratio:.2f}x above average")
                                else:
                                    output_text += f"  {channel}: {score:.6f} ✅ Normal (avg: {avg_score:.4f})\n"
                            else:
                                output_text += f"  {channel}: {score:.6f}\n"
                        else:
                            output_text += f"  {channel}: {score}\n"
                else:
                    output_text += f"  {scores}\n"
        
        # Add overall risk assessment
        if high_risk_warnings:
            output_text += "\n" + "="*50 + "\n"
            output_text += "🚨 RISK ASSESSMENT WARNINGS 🚨\n"
            output_text += "="*50 + "\n"
            for warning in high_risk_warnings:
                output_text += f"⚠️  {warning}\n"
            output_text += "\n📋 RECOMMENDATIONS:\n"
            output_text += "• Consider consulting with a healthcare professional\n"
            output_text += "• Monitor symptoms closely\n"
            output_text += "• Follow up with appropriate medical examinations\n"
        else:
            output_text += "\n✅ All projection scores are within normal ranges compared to positive case averages.\n"
        
        if 'Error' in results:
            output_text += f"\nError: {results['Error']}"
        
        return output_text

    def process_combined_file(self, combined_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine):
        """Process .pt file containing all signals"""
        try:
            if combined_file is None:
                return "Please upload a .pt file containing all signals"
            
            # Load .pt file
            pt_data = torch.load(combined_file.name, map_location='cpu')
            print('hello')
            if not isinstance(pt_data, dict):
                return "File format error, need a dictionary format .pt file"
            
            # Extract signal data
            eeg_data = None
            ecg_data = None
            resp_data = None
            
            if 'x' in pt_data:
                ecg_tensor = pt_data['x']
                ecg_data = ecg_tensor.numpy() if isinstance(ecg_tensor, torch.Tensor) else np.array(ecg_tensor)
                
            if 'x1' in pt_data:
                eeg_tensor = pt_data['x1']
                eeg_data = eeg_tensor.numpy() if isinstance(eeg_tensor, torch.Tensor) else np.array(eeg_tensor)
                
            if 'x3' in pt_data:
                resp_tensor = pt_data['x3']
                resp_data = resp_tensor.numpy() if isinstance(resp_tensor, torch.Tensor) else np.array(resp_tensor)
            
            # Check if at least one signal is present
            if eeg_data is None and ecg_data is None and resp_data is None:
                return "No valid signal data found in the file (x, x1, x3)"
            
            # Extract patient information
            extracted_patient_info = self.extract_patient_info_from_pt(pt_data)
            
            # Use extracted patient information (if available)
            if extracted_patient_info:
                final_age = extracted_patient_info.get('age', age)
                final_sex = extracted_patient_info.get('sex', sex)
                final_bmi = extracted_patient_info.get('bmi', bmi)
                final_chol = extracted_patient_info.get('chol', chol)
                final_hdl = extracted_patient_info.get('hdl', hdl)
                final_systbp = extracted_patient_info.get('systbp', systbp)
                final_parrptdiab = extracted_patient_info.get('parrptdiab', parrptdiab)
                final_smoke = extracted_patient_info.get('smoke', smoke)
                final_medicine = extracted_patient_info.get('medicine', medicine)
            else:
                final_age = age
                final_sex = sex
                final_bmi = bmi
                final_chol = chol
                final_hdl = hdl
                final_systbp = systbp
                final_parrptdiab = parrptdiab
                final_smoke = smoke
                final_medicine = medicine
            
            # Calculate projection scores
            results = self.calculate_projection_scores(
                eeg_data, ecg_data, resp_data,
                final_age, final_sex, final_bmi, final_chol, final_hdl, final_systbp, final_parrptdiab, final_smoke, final_medicine
            )
            print(results)
            
            # Format output with comparison and warnings
            detected_signals = []
            if eeg_data is not None:
                detected_signals.append("EEG")
            if ecg_data is not None:
                detected_signals.append("ECG") 
            if resp_data is not None:
                detected_signals.append("Resp")
            
            return self.format_results_with_comparison(
                results, 
                detected_signals=detected_signals, 
                extracted_patient_info=bool(extracted_patient_info),
                is_combined_mode=True
            )
            
        except Exception as e:
            return f"Error processing the unified .pt file: {str(e)}"

    def process_uploaded_files(self, eeg_file, ecg_file, resp_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine):
        """Process uploaded files"""
        try:
            # Read file data
            eeg_data = None
            ecg_data = None
            resp_data = None
            extracted_patient_info = None  # Used to store patient information extracted from .pt files
            
            if eeg_file is not None:
                if eeg_file.name.endswith('.npy'):
                    eeg_data = np.load(eeg_file.name)
                elif eeg_file.name.endswith('.npz'):
                    npz_data = np.load(eeg_file.name)
                    eeg_data = npz_data['EEG'] if 'EEG' in npz_data else npz_data[list(npz_data.keys())[0]]
                elif eeg_file.name.endswith('.txt'):
                    eeg_data = np.loadtxt(eeg_file.name)
                elif eeg_file.name.endswith('.pt'):
                    pt_data = torch.load(eeg_file.name, map_location='cpu')
                    if isinstance(pt_data, dict):
                        # If it's a dictionary format, try to get EEG data
                        if 'x1' in pt_data:
                            eeg_tensor = pt_data['x1']
                        elif 'EEG' in pt_data:
                            eeg_tensor = pt_data['EEG']
                        else:
                            return "EEG .pt file not found 'x1' or 'EEG' key"
                        
                        # Try to extract patient information
                        if extracted_patient_info is None:
                            extracted_patient_info = self.extract_patient_info_from_pt(pt_data)
                    else:
                        eeg_tensor = pt_data
                    eeg_data = eeg_tensor.numpy() if isinstance(eeg_tensor, torch.Tensor) else np.array(eeg_tensor)
                else:
                    return "EEG file format not supported, please use .npy, .npz, .txt or .pt format"
            
            if ecg_file is not None:
                if ecg_file.name.endswith('.npy'):
                    ecg_data = np.load(ecg_file.name)
                elif ecg_file.name.endswith('.npz'):
                    npz_data = np.load(ecg_file.name)
                    ecg_data = npz_data['ECG'] if 'ECG' in npz_data else npz_data[list(npz_data.keys())[0]]
                elif ecg_file.name.endswith('.txt'):
                    ecg_data = np.loadtxt(ecg_file.name)
                elif ecg_file.name.endswith('.pt'):
                    pt_data = torch.load(ecg_file.name, map_location='cpu')
                    if isinstance(pt_data, dict):
                        # If it's a dictionary format, try to get ECG data
                        if 'x' in pt_data:
                            ecg_tensor = pt_data['x']
                        elif 'ECG' in pt_data:
                            ecg_tensor = pt_data['ECG']
                        else:
                            return "ECG .pt file not found 'x' or 'ECG' key"
                        
                        # Try to extract patient information
                        if extracted_patient_info is None:
                            extracted_patient_info = self.extract_patient_info_from_pt(pt_data)
                    else:
                        ecg_tensor = pt_data
                    ecg_data = ecg_tensor.numpy() if isinstance(ecg_tensor, torch.Tensor) else np.array(ecg_tensor)
                else:
                    return "ECG file format not supported, please use .npy, .npz, .txt or .pt format"
            
            if resp_file is not None:
                if resp_file.name.endswith('.npy'):
                    resp_data = np.load(resp_file.name)
                elif resp_file.name.endswith('.npz'):
                    npz_data = np.load(resp_file.name)
                    resp_data = npz_data['Resp'] if 'Resp' in npz_data else npz_data[list(npz_data.keys())[0]]
                elif resp_file.name.endswith('.txt'):
                    resp_data = np.loadtxt(resp_file.name)
                elif resp_file.name.endswith('.pt'):
                    pt_data = torch.load(resp_file.name, map_location='cpu')
                    if isinstance(pt_data, dict):
                        # If it's a dictionary format, try to get Resp data
                        if 'x3' in pt_data:
                            resp_tensor = pt_data['x3']
                        elif 'Resp' in pt_data:
                            resp_tensor = pt_data['Resp']
                        else:
                            return "Resp .pt file not found 'x3' or 'Resp' key"
                        
                        # Try to extract patient information
                        if extracted_patient_info is None:
                            extracted_patient_info = self.extract_patient_info_from_pt(pt_data)
                    else:
                        resp_tensor = pt_data
                    resp_data = resp_tensor.numpy() if isinstance(resp_tensor, torch.Tensor) else np.array(resp_tensor)
                else:
                    return "Resp file format not supported, please use .npy, .npz, .txt or .pt format"
            
            # Check if at least one signal file is uploaded
            if eeg_data is None and ecg_data is None and resp_data is None:
                return "Please upload at least one signal file (EEG, ECG or Resp)"
            
            # If patient information is extracted from .pt files, use these information (otherwise use user input information)
            if extracted_patient_info:
                final_age = extracted_patient_info.get('age', age)
                final_sex = extracted_patient_info.get('sex', sex)
                final_bmi = extracted_patient_info.get('bmi', bmi)
                final_chol = extracted_patient_info.get('chol', chol)
                final_hdl = extracted_patient_info.get('hdl', hdl)
                final_systbp = extracted_patient_info.get('systbp', systbp)
                final_parrptdiab = extracted_patient_info.get('parrptdiab', parrptdiab)
                final_smoke = extracted_patient_info.get('smoke', smoke)
                final_medicine = extracted_patient_info.get('medicine', medicine)
            else:
                final_age = age
                final_sex = sex
                final_bmi = bmi
                final_chol = chol
                final_hdl = hdl
                final_systbp = systbp
                final_parrptdiab = parrptdiab
                final_smoke = smoke
                final_medicine = medicine
            
            # Calculate projection scores
            results = self.calculate_projection_scores(
                eeg_data, ecg_data, resp_data,
                final_age, final_sex, final_bmi, final_chol, final_hdl, final_systbp, final_parrptdiab, final_smoke, final_medicine
            )
            
            # Format output with comparison and warnings
            detected_signals = []
            if eeg_data is not None:
                detected_signals.append("EEG")
            if ecg_data is not None:
                detected_signals.append("ECG") 
            if resp_data is not None:
                detected_signals.append("Resp")
            
            return self.format_results_with_comparison(
                results, 
                detected_signals=detected_signals, 
                extracted_patient_info=bool(extracted_patient_info),
                is_combined_mode=False
            )
            
        except Exception as e:
            return f"Error processing the file: {str(e)}"

    def get_comparison_table(self):
        """Generate comparison table HTML"""
        html = """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 10px 0;">
            <h3 style="color: white; text-align: center; margin-bottom: 20px; font-size: 24px;">
                📊 Comparison Table of Positive Average Scores for Each Disease
            </h3>
            <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <table style="width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif;">
                    <thead>
                        <tr style="background: linear-gradient(90deg, #4CAF50, #45a049); color: white;">
                            <th style="padding: 12px; text-align: left; border-radius: 8px 0 0 0;">Disease Type</th>
                            <th style="padding: 12px; text-align: center;">ECG</th>
                            <th style="padding: 12px; text-align: center;">EEG</th>
                            <th style="padding: 12px; text-align: center; border-radius: 0 8px 0 0;">Resp</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
            # Add a row for each disease
        for disease_id, disease_name in self.disease_names.items():
            scores = self.positive_average_scores[disease_id]
            # Find the highest score for highlighting
            max_score = max(scores.values())
            
            # Alternate row colors
            bg_color = "#f8f9fa" if disease_id % 2 == 0 else "#ffffff"
            
            html += f"""
                        <tr style="background-color: {bg_color}; transition: background-color 0.3s;">
                            <td style="padding: 12px; font-weight: bold; color: #2c3e50;">{disease_name}</td>
            """
            
            for channel in ['ECG', 'EEG', 'Resp']:
                score = scores[channel]
                # Highlight the highest score
                if score == max_score:
                    style = "padding: 10px; text-align: center; background: linear-gradient(45deg, #FFD700, #FFA500); color: #8B4513; font-weight: bold; border-radius: 5px; margin: 2px;"
                else:
                    style = "padding: 10px; text-align: center; color: #34495e;"
                
                html += f'<td style="{style}">{score:.4f}</td>'
            
            html += "</tr>"
        
        html += """
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 15px; text-align: center;">
                <p style="color: white; font-size: 14px; margin: 5px 0;">
                    🌟 <strong>Golden Highlight</strong> indicates the highest score for this disease in all channels
                </p>
                <p style="color: #e8f4fd; font-size: 12px; margin: 0;">
                    The higher the value, the stronger the detection sensitivity of the channel corresponding to the disease
                </p>
            </div>
        </div>
        """
        
        return html

# Create Gradio interface
def create_gradio_interface():
    # Initialize inference class
    inference = GradioInference()
    
    with gr.Blocks(title="Physiological Signal Projection Score Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Physiological Signal Projection Score Analysis System")
        gr.Markdown("Upload EEG, ECG, Resp signal files and input patient information, the system will calculate the projection scores for various diseases")
        
        # Add feature selection tabs
        with gr.Tabs():
            # Analysis feature tab
            with gr.TabItem("🔍 Signal Analysis", elem_id="analysis_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Signal File Upload")
                        
                        # Select upload mode
                        upload_mode = gr.Radio(
                            choices=[("Upload Separately", "separate"), ("Upload Combined .pt File", "combined")], 
                            label="Upload Mode", 
                            value="separate"
                        )
                        
                        # Upload
                        with gr.Group(visible=True) as separate_group:
                            eeg_file = gr.File(label="EEG documents (.npy/.npz/.txt/.pt)", file_types=[".npy", ".npz", ".txt", ".pt"])
                            ecg_file = gr.File(label="ECG documents (.npy/.npz/.txt/.pt)", file_types=[".npy", ".npz", ".txt", ".pt"])
                            resp_file = gr.File(label="Resp documents (.npy/.npz/.txt/.pt)", file_types=[".npy", ".npz", ".txt", ".pt"])
                        
                        # Upload combined .pt file
                        with gr.Group(visible=False) as combined_group:
                            combined_file = gr.File(label="Combined .pt file containing all signals", file_types=[".pt"])
                        
                        gr.Markdown("### 👤 Patient Information")
                        with gr.Row():
                            age = gr.Number(label="Age", value=50, minimum=0, maximum=120)
                            sex = gr.Radio(choices=[("Male", 1), ("Female", 0)], label="Sex", value=1)
                        
                        with gr.Row():
                            bmi = gr.Number(label="BMI", value=25.0, minimum=10, maximum=50)
                            chol = gr.Number(label="Cholesterol (mg/dL)", value=200, minimum=100, maximum=400)
                        
                        with gr.Row():
                            hdl = gr.Number(label="HDL Cholesterol (mg/dL)", value=50, minimum=20, maximum=100)
                            systbp = gr.Number(label="Systolic Blood Pressure (mmHg)", value=120, minimum=80, maximum=200)
                        
                        with gr.Row():
                            parrptdiab = gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Diabetes", value=0)
                            smoke = gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Smoking", value=0)
                        
                        medicine = gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Taking Blood Pressure Medication", value=0)
                        
                        submit_btn = gr.Button("🔍 Analyze Projection Scores", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Analysis Results")
                        output = gr.Textbox(
                            label="Projection Score Results", 
                            lines=30, 
                            max_lines=50, 
                            placeholder="Click 'Analyze Projection Scores' button to view results..."
                        )
            
            # Comparison feature tab
            with gr.TabItem("📈 Comparison of Positive Average Scores", elem_id="comparison_tab"):
                gr.Markdown("### 🎯 Comparison of Positive Average Scores for Each Disease")
                gr.Markdown("This table shows the average performance of the model in detecting various diseases across different physiological signal channels")
                
                # Comparison table display area
                comparison_display = gr.HTML(
                    value=inference.get_comparison_table(),
                    label="Comparison Table"
                )
                
                # Add explanation information
                with gr.Accordion("📝 Comparison Data Explanation", open=True):
                    gr.Markdown("""
                    ### 📊 Data Interpretation:
                    
                    **🔹 Score Meaning:**
                    - Higher Value: Indicates stronger detection sensitivity of the channel corresponding to the disease
                    **🔹 Channel Characteristics:**
                    - **ECG (Electrocardiogram)**: Good detection performance for cardiovascular-related diseases (MI, RBBB, CHF, AFib)
                    - **EEG (Electroencephalography)**: Unique value for neurological-related cardiovascular risk assessment
                    - **Resp (Respiration)**: Helpful for overall pulmonary function assessment and long-term risk prediction
                    **⚠️ Note:** These are average values based on a large number of samples, and the actual scores for individual patients may differ
                    """)
                
                # Refresh button (optional)
                refresh_btn = gr.Button("🔄 Refresh Comparison Data", variant="secondary")
                refresh_btn.click(
                    fn=lambda: inference.get_comparison_table(),
                    outputs=comparison_display
                )
        
        # Usage instructions (moved outside the tab, as general instructions)
        with gr.Accordion("📖 Usage Instructions", open=False):
            gr.Markdown("""
            ### Supported File Formats:
            - **.npy**: NumPy array file
            - **.npz**: NumPy compressed file
            - **.txt**: Text file (one value per line)
            - **.pt**: PyTorch tensor file
            
            ### Upload Mode:
            1. **Upload Separately**: Upload EEG, ECG, Resp files separately
            2. **Upload Combined .pt File**: Upload a single .pt file containing all signals and patient information
            
            ### .pt File Format Requirements (Unified Mode):
            - **'x'**: ECG signal data
            - **'x1'**: EEG signal data
            - **'x3'**: Resp signal data
            - **'age'**: Patient age
            - **'sex'**: Patient sex
            - **'bmi'**: BMI index
            - **'chol'**: Cholesterol level
            - **'hdl'**: HDL cholesterol
            - **'systbp'**: Systolic blood pressure
            - **'parrptdiab'**: Diabetes status
            - **'smokstat_s1'**: Smoking status
            - **'htnmed1'**: Blood pressure medication use
            
            ### Signal Requirements:
            - **Sampling Rate**: 125 Hz for EEG,ECG and 10 Hz for Resp
            - **Duration**: 30 seconds
            - If the signal length is less than 30 seconds, the system will automatically pad with zeros
            - If the signal length is greater than 30 seconds, the system will truncate the first 30 seconds
            
            ### Disease Types:
            - **MI**: Myocardial Infarction
            - **RBBB**: Right Bundle Branch Block
            - **CHF**: Congestive Heart Failure
            - **Hypertension**: Hypertension
            - **Future Hypertension**: Future Hypertension Risk
            - **Future Death**: Future CVD (Cardiovascular Disease) Death Risk
            - **AFib**: Atrial Fibrillation
            - **Incident AFib**: Incident Atrial Fibrillation
            
            ### Notes:
            - At least one signal file (EEG, ECG or Resp) must be uploaded
            - The higher the projection score, the greater the risk of the disease
            - The system will automatically calculate the FRS (Framingham Risk Score) cardiovascular risk score
            
            ### Risk Assessment Features:
            - **🔴 HIGH RISK**: Scores >1.5x the positive case average (immediate attention recommended)
            - **🟡 ELEVATED**: Scores above positive case average but <1.5x (monitoring recommended)  
            - **✅ Normal**: Scores within or below positive case average ranges
            - The system automatically compares your scores with established positive case averages
            - Risk warnings include specific recommendations for follow-up care
            """)
        
        # 模式切换逻辑
        def toggle_upload_mode(mode):
            if mode == "separate":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        upload_mode.change(
            fn=toggle_upload_mode,
            inputs=[upload_mode],
            outputs=[separate_group, combined_group]
        )
        
        # Process function selector
        def process_files(upload_mode, eeg_file, ecg_file, resp_file, combined_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine):
            if upload_mode == "separate":
                return inference.process_uploaded_files(eeg_file, ecg_file, resp_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine)
            else:
                return inference.process_combined_file(combined_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine)
        
        # Bind events
        submit_btn.click(
            fn=process_files,
            inputs=[upload_mode, eeg_file, ecg_file, resp_file, combined_file, age, sex, bmi, chol, hdl, systbp, parrptdiab, smoke, medicine],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 