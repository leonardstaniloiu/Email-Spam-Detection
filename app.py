import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime


from spam_service import (
    generate_random_email,
    load_data,
    get_all_models,
    train_all_models,
    predict_email,
)

def main():

    st.set_page_config(
        page_title="SpamX - Email spam detector",
        page_icon="🤖",
        layout="wide"
    )
    
    if 'email_text' not in st.session_state:
        st.session_state.email_text = "WINNER!! You have won a $100 gift card. Click here NOW to claim your prize!!!"
    
    st.title(" Spam email detector - using multiple ML algorithms")
    
    data_path = "dataset/spam.csv"
    
    with st.spinner("Training all models... This may take a moment!"):
        vectorizer, results, df, X_test, y_test = train_all_models(data_path)
    
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("All emails", len(df))
    with col2:
        spam_count = df["label"].sum()
        st.metric("Spam", spam_count, delta=f"{spam_count/len(df)*100:.1f}%")
    with col3:
        ham_count = len(df) - spam_count
        st.metric("Legit", ham_count, delta=f"{ham_count/len(df)*100:.1f}%")
    with col4:
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        st.metric("Best Model", best_model[0][:15], delta=f"{best_model[1]['accuracy']:.1%}")
    
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Test an Email",
        "📊 Model Comparison",
        "🎯 Confusion Matrix"
    ])
    
    with tab1:
        
        col1, col2 = st.columns([2, 1])
        
        with col1:

            if st.button("Generate Random Email"):
                st.session_state.email_text = generate_random_email()

            email_text = st.text_area(
                "Your email text here:",
                value=st.session_state.email_text,
                height=150
            )
            
            st.session_state.email_text = email_text
        
        with col2:
            st.write("**Choose algorithm:**")
            model_choice = st.radio(
                "algorithm:",
                list(results.keys()),
                label_visibility="collapsed"
            )
            
        if st.button(" Analyze email", type="primary", use_container_width=False):
            if email_text.strip():
                model = results[model_choice]["model"]
                prediction, probability = predict_email(email_text, vectorizer, model)
                
                st.write("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("### SPAM DETECTED!")
                        st.write(f"**Model used:** {model_choice}")
                        if probability:
                            st.write(f"**Confidence:** {probability:.1%}")
                    else:
                        st.success("### Email LEGIT")
                        st.write(f"**Model used:** {model_choice}")
                        if probability:
                            st.write(f"**Spam probability:** {probability:.1%}")
                
                with col2:
                    st.write(f"**Model accuracy:** {results[model_choice]['accuracy']:.1%}")
            else:
                st.warning("Write an email text to analyze!")
    
    with tab2:
        st.subheader("Model Comparison")
        
        comparison_data = []
        for name, data in results.items():
            comparison_data.append({
                "Algorithm": name,
                "Accuracy": data["accuracy"],
                "Precision": data["precision"],
                "Recall": data["recall"],
                "F1-Score": data["f1"],
                "Training time in seconds": data["train_time"]
            })
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values("Accuracy", ascending=False)
        
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "Training time in seconds": "{:.2f}"
            }).background_gradient(subset=["Accuracy"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                comp_df,
                x="Accuracy",
                y="Algorithm",
                orientation='h',
                title="Accuracy on Test Data",
                color="Accuracy",
                color_continuous_scale="RdYlGn"
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                comp_df,
                x="Training time in seconds",
                y="Algorithm",
                orientation='h',
                title="Training Time",
                color="Training time in seconds",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.scatter(
            comp_df,
            x="Precision",
            y="Recall",
            text="Algorithm",
            size="F1-Score",
            color="F1-Score",
            title="Precision vs Recall",
            size_max=30
        )
        fig3.update_traces(textposition='top center')
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Confusion Matrix")
        
        cols = st.columns(2)
        
        for idx, (name, data) in enumerate(results.items()):
            with cols[idx % 2]:
                cm = data["confusion_matrix"]
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Ham (Predicted)', 'Spam (Predicted)'],
                    y=['Ham (Actual)', 'Spam (Actual)'],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig.update_layout(
                    title=f"{name} (Acc: {data['accuracy']:.1%})",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
   
if __name__ == "__main__":
    main()