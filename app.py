import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from PIL import Image

def main():
    image= Image.open("Images/DataDoctor.png")
    st.image(image, use_column_width=True)
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    query = st.text_input("Enter your query:")

    if st.button("Run Query"):
        if openai_api_key:
            agent = create_csv_agent(
                OpenAI(
                    openai_api_key=openai_api_key,
                    temperature=0,
                ),
                "datapoint.csv",
                verbose=True,
            )
            result = agent.run(query)
            st.success(f"Result: {result}")
            
        else:
            st.error("Please enter your OpenAI API Key.")

    # Add some more text to the page
    st.header("Welcome to the DataDoctor!")
    st.write("This app allows you to query an OpenAI model and get back the results in CSV format.")
    st.write("To get started, enter your OpenAI API key and a query, then click the 'Run Query' button.")
    st.write("The results will be displayed when you click the ""Run Query"" button.")
    st.write("Good luck!")

if __name__ == "__main__":
    main()
