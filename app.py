import gradio as gr
import pandas as pd
import pickle
import numpy as np

#Load the Model
with open("VG_rf_pipeline.pkl", "rb") as file:
   model = pickle.load(file)
 
#main logic
def predict_sales(Name,Platform,Year,Genre,Publisher,NA_Sales,EU_Sales,JP_Sales,Other_Sales):

    
    input_df = pd.DataFrame([[

        Name,Platform,Year,Genre,Publisher,NA_Sales,EU_Sales,JP_Sales,Other_Sales

    ]],
    columns =  [
        'Name','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales'
    ]
    )


    prediction = model.predict(input_df)[0]
    return prediction

#interface

app = gr.Interface(
    fn = predict_sales,
    inputs=[
        gr.Textbox(label="Name"), 
        gr.Dropdown(["PS4", "Xbox", "PC","PS3","PS2"], label="Platform"), 
        gr.Slider(1980, 2020, step=1, label="Year"), 
        gr.Dropdown(["Action", "Sports", "RPG", "Shooter","Platform","Racing","Role-Playing","Puzzle","Misc","Shooter","Simulation","Action","Fighting","Adventure","Strategy"], label="Genre"), 
        gr.Textbox(label="Publisher"), 
        gr.Number(label="NA Sales"), 
        gr.Number(label="EU Sales"), 
        gr.Number(label="JP Sales"),
        gr.Number(label="Other Sales")
        ],
    outputs = "text",
    title = "Video Games sales prediction"
)

#launch 
app.launch(share=True) 