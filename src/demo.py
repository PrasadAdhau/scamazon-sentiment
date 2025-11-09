import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class AmazonSentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scam-azon? - Amazon Review Sentiment Analyzer")
        self.root.geometry("1000x750")
        self.root.configure(bg="#FFFFFF")
        
        # Initialize the sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
        
        # Initialize current review tracker
        self.current_review = None
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model paths
        self.model_paths = {
            'Books': {
                'BERT': 'Books_bert_model/checkpoint-6195/',
                'DistilBERT': 'Books_distilbert_model/checkpoint-6195/',
                'Logistic Regression': 'Books_logistic_regression_model.joblib'
            },
            'Electronics': {
                'BERT': 'Electronics_bert_model/checkpoint-5901/',
                'DistilBERT': 'Electronics_distilbert_model/checkpoint-5901/',
                'Logistic Regression': 'Electronics_logistic_regression_model.joblib'
            }
        }
        
        # Initialize base model paths
        self.base_model_paths = {
            'BERT': 'bert-base-uncased',
            'DistilBERT': 'distilbert-base-uncased'
        }
        
        # Initialize models dictionary (lazy loading)
        self.models = {}
        self.tokenizers = {}
        self.base_models = {}
        self.base_tokenizers = {}
        
        # Create and place the widgets
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
    def load_dataset(self):
        """Load Amazon review datasets"""
        self.books_df = None
        self.electronics_df = None
        
        # Attempt to load datasets
        try:
            print("Loading Books dataset...")
            self.books_df = pd.read_csv('data/amazon_reviews_us_Books_v1_02.tsv',
                                       sep='\t',
                                       on_bad_lines='skip',
                                       quoting=3,
                                       encoding='utf-8',
                                       engine='python',
                                       nrows=10000)  # Only read 10k rows
            
            print(f"Successfully loaded Books dataset with {len(self.books_df)} rows")
            self.books_df['category'] = 'Books'
        except Exception as e:
            print(f"Error loading Books dataset: {e}")
            self.books_df = pd.DataFrame()
            
        try:
            print("Loading Electronics dataset...")
            self.electronics_df = pd.read_csv('data/amazon_reviews_us_Electronics_v1_00.tsv',
                                             sep='\t',
                                             on_bad_lines='skip',
                                             quoting=3,
                                             encoding='utf-8',
                                             engine='python',
                                             nrows=10000)  # Only read 10k rows
            
            print(f"Successfully loaded Electronics dataset with {len(self.electronics_df)} rows")
            self.electronics_df['category'] = 'Electronics'
        except Exception as e:
            print(f"Error loading Electronics dataset: {e}")
            self.electronics_df = pd.DataFrame()
        
        # Process datasets - add sentiment labels
        if not self.books_df.empty:
            self.books_df = self.add_sentiment_labels(self.books_df)
            self.books_df = self.add_helpfulness_ratio(self.books_df)
            self.books_df['aspects'] = self.books_df['review_body'].apply(self.extract_aspects)
        
        if not self.electronics_df.empty:
            self.electronics_df = self.add_sentiment_labels(self.electronics_df)
            self.electronics_df = self.add_helpfulness_ratio(self.electronics_df)
            self.electronics_df['aspects'] = self.electronics_df['review_body'].apply(self.extract_aspects)
    
    def add_sentiment_labels(self, df):
        """Convert star_rating to categorical sentiment"""
        # First ensure star_rating is numeric
        df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
        df['star_rating'] = df['star_rating'].fillna(0)
        
        # Convert to sentiment labels
        df['sentiment'] = df['star_rating'].apply(
            lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive')
        )
        return df
    
    def add_helpfulness_ratio(self, df):
        """Create helpfulness ratio for analysis"""
        # Ensure vote columns are numeric
        df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce')
        df['total_votes'] = pd.to_numeric(df['total_votes'], errors='coerce')
        df['helpful_votes'] = df['helpful_votes'].fillna(0)
        df['total_votes'] = df['total_votes'].fillna(0)
        
        # Avoid division by zero
        df['helpfulness_ratio'] = df.apply(
            lambda row: row['helpful_votes'] / row['total_votes'] if row['total_votes'] > 0 else 0,
            axis=1
        )
        return df
    
    def extract_aspects(self, text):
        """Extract aspects from review text"""
        if pd.isna(text):
            return []
        
        # Simple rule-based aspect extraction
        aspects = []
        
        # Common product aspects
        book_aspects = ["plot", "character", "writing", "author", "story", "ending", "price"]
        electronics_aspects = ["battery", "screen", "quality", "price", "durability", "speed", "performance"]
        
        # Check for these aspects in the text
        for aspect in book_aspects + electronics_aspects:
            if aspect.lower() in text.lower():
                aspects.append(aspect)
        
        return aspects
        
    def get_model(self, category, model_type, base_model=False):
        """Lazy-load and return the selected model"""
        model_key = f"{category}_{model_type}"
        
        if base_model:
            model_key = f"base_{model_type}"
            
            # Return cached base model if already loaded
            if model_key in self.base_models:
                return self.base_models[model_key], self.base_tokenizers[model_key]
            
            # Load base model
            try:
                base_path = self.base_model_paths[model_type]
                print(f"Loading base model from {base_path}")
                
                if model_type == 'BERT':
                    tokenizer = BertTokenizer.from_pretrained(base_path)
                    model = BertForSequenceClassification.from_pretrained(base_path, num_labels=3)
                elif model_type == 'DistilBERT':
                    tokenizer = DistilBertTokenizer.from_pretrained(base_path)
                    model = DistilBertForSequenceClassification.from_pretrained(base_path, num_labels=3)
                else:
                    return None, None
                
                self.base_models[model_key] = model
                self.base_tokenizers[model_key] = tokenizer
                
                return model, tokenizer
            except Exception as e:
                print(f"Error loading base model: {e}")
                messagebox.showerror("Model Error", f"Failed to load base {model_type} model. Using VADER instead.")
                return None, None
        else:
            # If model type is Logistic Regression, we'll use VADER
            if model_type == "Logistic Regression":
                return None, None
            
            # Return cached model if already loaded
            if model_key in self.models:
                return self.models[model_key], self.tokenizers[model_key]
            
            # Load model from path
            try:
                model_path = self.model_paths[category][model_type]
                print(f"Loading finetuned model from {model_path}")
                
                # Load the appropriate tokenizer based on model type
                if model_type == 'BERT':
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    model = BertForSequenceClassification.from_pretrained(model_path)
                elif model_type == 'DistilBERT':
                    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                    model = DistilBertForSequenceClassification.from_pretrained(model_path)
                else:
                    return None, None
                
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                
                return model, tokenizer
            except Exception as e:
                print(f"Error loading model: {e}")
                messagebox.showerror("Model Error", f"Failed to load {model_type} model for {category}. Using VADER instead.")
                return None, None
    
    def create_header(self):
        """Create Amazon-like header with logo and navigation"""
        header_frame = tk.Frame(self.root, bg="#232F3E", height=60)
        header_frame.pack(fill=tk.X)
        
        # Amazon-like logo
        logo_label = tk.Label(header_frame, text="scam-azon?", font=("Arial", 24, "bold"), 
                             fg="#FF9900", bg="#232F3E", padx=20)
        logo_label.pack(side=tk.LEFT, pady=10)
        
        # Subtitle 
        subtitle = tk.Label(header_frame, text="Review Sentiment Analyzer", 
                           font=("Arial", 14), fg="white", bg="#232F3E")
        subtitle.pack(side=tk.LEFT, padx=10, pady=15)
        
    def create_main_content(self):
        """Create the main content area with controls and results"""
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left column for controls
        control_frame = tk.Frame(main_frame, bg="white", width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(control_frame, text="Analyze Amazon Reviews", 
                              font=("Arial", 16, "bold"), bg="white")
        title_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Category selection
        category_frame = tk.Frame(control_frame, bg="white")
        category_frame.pack(fill=tk.X, pady=5)
        
        category_label = tk.Label(category_frame, text="Category:", font=("Arial", 12), bg="white")
        category_label.pack(side=tk.LEFT)
        
        self.category_var = tk.StringVar(value="Books")
        category_menu = ttk.Combobox(category_frame, textvariable=self.category_var, 
                                    values=["Books", "Electronics"], state="readonly", width=15)
        category_menu.pack(side=tk.LEFT, padx=(10, 0))
        
        # Model selection
        model_frame = tk.Frame(control_frame, bg="white")
        model_frame.pack(fill=tk.X, pady=5)
        
        model_label = tk.Label(model_frame, text="Model:", font=("Arial", 12), bg="white")
        model_label.pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar(value="BERT")
        model_menu = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                values=["BERT", "DistilBERT", "Logistic Regression"], state="readonly", width=15)
        model_menu.pack(side=tk.LEFT, padx=(10, 0))
        
        # Sentiment selection for random review
        sentiment_frame = tk.Frame(control_frame, bg="white")
        sentiment_frame.pack(fill=tk.X, pady=5)
        
        sentiment_label = tk.Label(sentiment_frame, text="Load Random:", font=("Arial", 12), bg="white")
        sentiment_label.pack(side=tk.LEFT)
        
        self.sentiment_var = tk.StringVar(value="Any")
        sentiment_menu = ttk.Combobox(sentiment_frame, textvariable=self.sentiment_var, 
                                    values=["Any", "Positive", "Neutral", "Negative"], state="readonly", width=15)
        sentiment_menu.pack(side=tk.LEFT, padx=(10, 0))
        
        # Random review button
        random_review_btn = tk.Button(control_frame, text="Load Random Review", 
                                    bg="#F0F2F2", activebackground="#E7E9EC",
                                    command=self.load_random_review, height=2)
        random_review_btn.pack(fill=tk.X, pady=10)
        
        # Review text input
        text_label = tk.Label(control_frame, text="Review Text:", font=("Arial", 12), bg="white")
        text_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.review_text = scrolledtext.ScrolledText(control_frame, wrap=tk.WORD, height=10)
        self.review_text.pack(fill=tk.X, pady=(0, 10))
        
        # Analyze button (Amazon-style)
        analyze_btn = tk.Button(control_frame, text="Analyze Sentiment", font=("Arial", 12, "bold"),
                             bg="#FF9900", activebackground="#FFAC31", fg="black",
                             command=self.analyze_sentiment, height=2)
        analyze_btn.pack(fill=tk.X, pady=10)
        
        # Right columns for results
        self.results_frame_left = tk.Frame(main_frame, bg="#F9F9F9", width=275)
        self.results_frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_frame_right = tk.Frame(main_frame, bg="#F9F9F9", width=275)
        self.results_frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.results_title_left = tk.Label(self.results_frame_left, text="Finetuned Model Results", 
                                        font=("Arial", 16, "bold"), bg="#F9F9F9")
        self.results_title_left.pack(anchor=tk.W, padx=20, pady=10)
        
        self.results_title_right = tk.Label(self.results_frame_right, text="Base Model Results", 
                                         font=("Arial", 16, "bold"), bg="#F9F9F9")
        self.results_title_right.pack(anchor=tk.W, padx=20, pady=10)
        
        # Initially hide results content
        self.results_content_left = tk.Frame(self.results_frame_left, bg="#F9F9F9")
        self.results_content_left.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.results_content_left.pack_forget()  # Hide initially
        
        self.results_content_right = tk.Frame(self.results_frame_right, bg="#F9F9F9")
        self.results_content_right.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.results_content_right.pack_forget()  # Hide initially
        
    def create_footer(self):
        """Create Amazon-like footer"""
        footer_frame = tk.Frame(self.root, bg="#232F3E", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_text = tk.Label(footer_frame, text="ISTE-612: Information Retrieval and Text Mining | Spring 2025", 
                              fg="white", bg="#232F3E", font=("Arial", 10))
        footer_text.pack(pady=5)
    
    def load_random_review(self):
        """Load a random review from the selected category with balanced sentiment distribution"""
        category = self.category_var.get()
        requested_sentiment = self.sentiment_var.get().lower()
        
        if category == "Books":
            df = self.books_df
        else:
            df = self.electronics_df
        
        if df.empty:
            messagebox.showerror("Error", f"No {category} dataset loaded or dataset is empty")
            return
        
        # Select a random review with specified sentiment
        try:
            # Filter reviews with non-empty review body
            filtered_df = df[df['review_body'].notna() & (df['review_body'] != '')]
            
            if filtered_df.empty:
                messagebox.showerror("Error", f"No valid reviews found in {category} dataset")
                return
            
            # Filter by sentiment if specified
            if requested_sentiment != "any":
                sentiment_df = filtered_df[filtered_df['sentiment'] == requested_sentiment]
                
                if sentiment_df.empty:
                    messagebox.showwarning("Warning", f"No {requested_sentiment} reviews found. Loading any review instead.")
                    sentiment_df = filtered_df
            else:
                # For "Any", get a balanced sample by first selecting sentiment
                # Get counts of each sentiment
                sentiment_counts = filtered_df['sentiment'].value_counts()
                available_sentiments = sentiment_counts.index.tolist()
                
                # Randomly select a sentiment type with equal probability
                selected_sentiment = random.choice(['positive', 'neutral', 'negative'])
                
                # If the selected sentiment is not available, choose a random one from what is available
                if selected_sentiment not in available_sentiments:
                    selected_sentiment = random.choice(available_sentiments)
                
                sentiment_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
            
            # Get a random review from the filtered set
            random_review = sentiment_df.sample(1).iloc[0]
            
            # Store the current review for reference
            self.current_review = random_review
            
            # Set the review text
            self.review_text.delete(1.0, tk.END)
            self.review_text.insert(tk.END, random_review['review_body'])
            
            # Set additional info in title or footer
            review_info = f"Product: {random_review.get('product_title', 'N/A')[:50]}... | "
            review_info += f"Rating: {random_review.get('star_rating', 'N/A')} | "
            review_info += f"Sentiment: {random_review.get('sentiment', 'N/A').capitalize()} | "
            review_info += f"Votes: {int(random_review.get('helpful_votes', 0))}/{int(random_review.get('total_votes', 0))}"
            
            # Display info
            messagebox.showinfo("Random Review Info", review_info)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load random review: {str(e)}")
    
    def extract_aspects_for_analysis(self, text, category):
        """Extract aspects from review text based on category for analysis display"""
        aspects = {}
        
        # Define category-specific aspects
        book_aspects = ["plot", "character", "writing", "author", "story", "ending", "price"]
        electronics_aspects = ["battery", "screen", "quality", "price", "durability", "speed", "performance"]
        
        # Choose appropriate aspect list based on category
        aspect_list = book_aspects if category.lower() == 'books' else electronics_aspects
        
        # Check for aspects in the text
        sentences = re.split(r'[.!?]', text)
        
        for aspect in aspect_list:
            aspect_scores = []
            for sentence in sentences:
                if aspect.lower() in sentence.lower():
                    score = self.sid.polarity_scores(sentence)['compound']
                    aspect_scores.append(score)
            
            # If we found the aspect in any sentence
            if aspect_scores:
                avg_score = sum(aspect_scores) / len(aspect_scores)
                aspects[aspect] = {
                    'score': avg_score,
                    'sentiment': 'positive' if avg_score > 0.1 else ('negative' if avg_score < -0.1 else 'neutral')
                }
                
        return aspects
    
    def predict_with_transformer(self, text, model, tokenizer):
        """Get sentiment prediction using transformer model"""
        try:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get prediction
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            probabilities = probabilities.detach().numpy()[0]
            
            # Map indices to labels (assuming 3-class classification: negative, neutral, positive)
            if len(probabilities) == 3:
                result = {
                    'negative': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'positive': float(probabilities[2])
                }
            elif len(probabilities) == 2:
                # Binary classification
                result = {
                    'negative': float(probabilities[0]),
                    'positive': float(probabilities[1]),
                    'neutral': 0.0  # Set neutral to 0 for binary models
                }
            
            # Determine sentiment
            sentiment = max(result, key=result.get)
            confidence = result[sentiment]
            
            return sentiment, confidence, result
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            # Fallback to VADER
            return self.predict_with_vader(text)
    
    def predict_with_vader(self, text):
        """Get sentiment prediction using VADER"""
        # VADER sentiment analysis
        sentiment_scores = self.sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        # Map to categorical sentiment
        if compound_score > 0.2:
            sentiment = 'positive'
            confidence = min(0.5 + compound_score/2, 0.99)
        elif compound_score < -0.2:
            sentiment = 'negative'
            confidence = min(0.5 + abs(compound_score)/2, 0.99)
        else:
            sentiment = 'neutral'
            confidence = 0.5 + abs(compound_score)/4
        
        # Calculate probabilities for visualization
        if sentiment == 'positive':
            probabilities = {
                'positive': confidence,
                'neutral': (1 - confidence) * 0.7,
                'negative': (1 - confidence) * 0.3
            }
        elif sentiment == 'negative':
            probabilities = {
                'negative': confidence,
                'neutral': (1 - confidence) * 0.7,
                'positive': (1 - confidence) * 0.3
            }
        else:
            probabilities = {
                'neutral': confidence,
                'positive': (1 - confidence) * 0.6,
                'negative': (1 - confidence) * 0.4
            }
        
        return sentiment, confidence, probabilities
    
    def display_sentiment_results(self, frame, sentiment, confidence, probabilities, star_rating=None, model_name=""):
        """Display sentiment analysis results in the specified frame"""
        for widget in frame.winfo_children():
            widget.destroy()
            
        # Colors for sentiment display
        colors = {
            'positive': '#007600',  # Amazon green
            'neutral': '#C45500',   # Amazon orange
            'negative': '#B12704'   # Amazon red
        }
        
        # Overall sentiment
        sentiment_frame = tk.Frame(frame, bg="#F9F9F9")
        sentiment_frame.pack(fill=tk.X, pady=10)
        
        sentiment_label = tk.Label(sentiment_frame, text=f"{model_name}\nOverall Sentiment: ", 
                                  font=("Arial", 12), bg="#F9F9F9")
        sentiment_label.pack(anchor=tk.W)
        
        sentiment_value = tk.Label(sentiment_frame, text=sentiment.upper(), 
                                 font=("Arial", 14, "bold"), fg=colors[sentiment], bg="#F9F9F9")
        sentiment_value.pack(anchor=tk.W)
        
        confidence_label = tk.Label(sentiment_frame, text=f"Confidence: {confidence:.2f}", 
                                   font=("Arial", 12), bg="#F9F9F9")
        confidence_label.pack(anchor=tk.W)
        
        # Display star rating if available
        if star_rating is not None:
            star_frame = tk.Frame(frame, bg="#F9F9F9")
            star_frame.pack(fill=tk.X, pady=5)
            
            star_label = tk.Label(star_frame, text=f"Actual Star Rating: ", 
                                 font=("Arial", 12), bg="#F9F9F9")
            star_label.pack(anchor=tk.W)
            
            # Determine color based on star rating
            star_color = '#007600' if star_rating >= 4 else ('#C45500' if star_rating >= 3 else '#B12704')
            
            star_value = tk.Label(star_frame, text=f"{star_rating} / 5", 
                                 font=("Arial", 14, "bold"), fg=star_color, bg="#F9F9F9")
            star_value.pack(anchor=tk.W)
            
            # Add a match/mismatch indicator
            predicted_positive = sentiment == 'positive'
            predicted_negative = sentiment == 'negative'
            actual_positive = star_rating >= 4
            actual_negative = star_rating <= 2
            
            if (predicted_positive and actual_positive) or (predicted_negative and actual_negative) or (sentiment == 'neutral' and star_rating == 3):
                match_label = tk.Label(star_frame, text="Match ✓", 
                                     font=("Arial", 12), fg="#007600", bg="#F9F9F9")
            else:
                match_label = tk.Label(star_frame, text="Mismatch ✗", 
                                     font=("Arial", 12), fg="#B12704", bg="#F9F9F9")
            match_label.pack(anchor=tk.W)
        
        # Sentiment chart
        fig = plt.Figure(figsize=(3, 2), dpi=100)
        ax = fig.add_subplot(111)
        
        # Make sure the order is consistent for the bars (negative, neutral, positive)
        ordered_sentiments = ['negative', 'neutral', 'positive']
        ordered_probs = []
        for s in ordered_sentiments:
            if s in probabilities:
                ordered_probs.append(probabilities[s])
            else:
                ordered_probs.append(0.0)
        
        bars = ax.bar(ordered_sentiments, ordered_probs, 
                     color=[colors['negative'], colors['neutral'], colors['positive']])
        
        ax.set_ylim(0, 1)
        ax.set_title('Sentiment Probabilities')
        ax.set_yticks([0, 0.5, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add probability values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        chart_frame = tk.Frame(frame, bg="#F9F9F9")
        chart_frame.pack(fill=tk.X, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Make frame visible
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def display_aspect_analysis(self, aspects, frame):
        """Display aspect-based sentiment analysis in the specified frame"""
        if not aspects:
            return
            
        aspect_label = tk.Label(frame, text="Aspect-Based Sentiment:", 
                               font=("Arial", 12), bg="#F9F9F9")
        aspect_label.pack(anchor=tk.W, pady=(10, 5))
        
        # Colors for sentiment display
        colors = {
            'positive': '#007600',  # Amazon green
            'neutral': '#C45500',   # Amazon orange
            'negative': '#B12704'   # Amazon red
        }
        
        # Create a frame for the aspect table
        aspect_frame = tk.Frame(frame, bg="#F9F9F9")
        aspect_frame.pack(fill=tk.X, pady=5)
        
        # Create a table-like display
        headers = ["Aspect", "Sentiment"]
        for i, header in enumerate(headers):
            header_label = tk.Label(aspect_frame, text=header, font=("Arial", 10, "bold"), 
                                  bg="#E6E6E6", width=10, relief=tk.RIDGE, padx=2, pady=2)
            header_label.grid(row=0, column=i, sticky="ew")
        
        # Add aspect rows
        for i, (aspect, data) in enumerate(aspects.items(), 1):
            aspect_name = tk.Label(aspect_frame, text=aspect.capitalize(), font=("Arial", 10), 
                                 bg="#F9F9F9", width=10, padx=2, pady=2, relief=tk.RIDGE)
            aspect_name.grid(row=i, column=0, sticky="ew")
            
            sentiment_text = tk.Label(aspect_frame, text=data['sentiment'].upper(), font=("Arial", 10),
                                   fg=colors[data['sentiment']], bg="#F9F9F9", width=10, 
                                   padx=2, pady=2, relief=tk.RIDGE)
            sentiment_text.grid(row=i, column=1, sticky="ew")
    
    def display_highlighted_review(self, text, aspects, frame):
        """Display the review with highlighted aspects"""
        # Highlighted Review Text
        review_label = tk.Label(frame, text="Review with Highlighted Aspects:", 
                              font=("Arial", 12), bg="#F9F9F9")
        review_label.pack(anchor=tk.W, pady=(10, 5))
        
        # Create a text widget for the review with highlighted aspects
        highlighted_text = tk.Text(frame, wrap=tk.WORD, height=5, 
                                 bg="white", relief=tk.RIDGE, padx=5, pady=5)
        highlighted_text.pack(fill=tk.X, pady=5)
        
        # Add the review text to the widget
        highlighted_text.insert(tk.END, text)
        
        # Colors for sentiment display
        colors = {
            'positive': '#007600',  # Amazon green
            'neutral': '#C45500',   # Amazon orange
            'negative': '#B12704'   # Amazon red
        }
        
        # Highlight aspects if any were found
        if aspects:
            for aspect in aspects.keys():
                start_idx = '1.0'
                while True:
                    # Find the aspect in the text
                    idx = highlighted_text.search(aspect, start_idx, tk.END, nocase=1)
                    if not idx:
                        # If the aspect is not found (anymore), break the loop
                        break
                    
                    # Calculate the end index
                    end_idx = f"{idx}+{len(aspect)}c"
                    
                    # Get the sentiment for this aspect
                    sentiment = aspects[aspect]['sentiment']
                    
                    # Add tags to highlight the aspect
                    highlighted_text.tag_add(sentiment, idx, end_idx)
                    
                    # Set the next start index
                    start_idx = end_idx
            
            # Configure tags with colors
            highlighted_text.tag_configure('positive', foreground=colors['positive'], font=("Arial", 9, "bold"))
            highlighted_text.tag_configure('neutral', foreground=colors['neutral'], font=("Arial", 9, "bold"))
            highlighted_text.tag_configure('negative', foreground=colors['negative'], font=("Arial", 9, "bold"))
        
        # Make the text widget read-only
        highlighted_text.configure(state=tk.DISABLED)
    
    def analyze_sentiment(self):
        """Analyze sentiment of the review text using both finetuned and base models"""
        text = self.review_text.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter a review text to analyze")
            return
        
        category = self.category_var.get()
        model_type = self.model_var.get()
        
        # Track if this is a review from the dataset with a star_rating
        star_rating = None
        
        # Try to find the star rating if this is a random review from the dataset
        if hasattr(self, 'current_review') and self.current_review is not None:
            if 'star_rating' in self.current_review:
                star_rating = self.current_review['star_rating']
        
        # Clear previous results
        for widget in self.results_content_left.winfo_children():
            widget.destroy()
        
        for widget in self.results_content_right.winfo_children():
            widget.destroy()
        
        # Extract aspect-based sentiment (always use VADER for this regardless of model choice)
        aspects = self.extract_aspects_for_analysis(text, category)
        
        # Use appropriate model for prediction
        try:
            # Status message for finetuned model
            self.results_title_left.config(text=f"Analyzing with Finetuned {model_type}...")
            self.root.update()
            
            # Status message for base model
            self.results_title_right.config(text=f"Analyzing with Base {model_type}...")
            self.root.update()
            
            # Analyze with finetuned model
            if model_type == "Logistic Regression":
                # Use VADER for Logistic Regression
                finetuned_sentiment, finetuned_confidence, finetuned_probabilities = self.predict_with_vader(text)
                finetuned_model_name = "Logistic Regression (VADER)"
            else:
                # Load the finetuned model
                finetuned_model, finetuned_tokenizer = self.get_model(category, model_type)
                
                if finetuned_model is None:
                    # Fallback to VADER if model loading failed
                    finetuned_sentiment, finetuned_confidence, finetuned_probabilities = self.predict_with_vader(text)
                    finetuned_model_name = f"Finetuned {model_type} (fallback to VADER)"
                else:
                    # Use transformer model
                    finetuned_sentiment, finetuned_confidence, finetuned_probabilities = self.predict_with_transformer(text, finetuned_model, finetuned_tokenizer)
                    finetuned_model_name = f"Finetuned {model_type}"
            
            # Analyze with base model
            if model_type == "Logistic Regression":
                # Use VADER for Logistic Regression
                base_sentiment, base_confidence, base_probabilities = self.predict_with_vader(text)
                base_model_name = "VADER"
            else:
                # Load the base model
                base_model, base_tokenizer = self.get_model(category, model_type, base_model=True)
                
                if base_model is None:
                    # Fallback to VADER if model loading failed
                    base_sentiment, base_confidence, base_probabilities = self.predict_with_vader(text)
                    base_model_name = f"Base {model_type} (fallback to VADER)"
                else:
                    # Use transformer model
                    base_sentiment, base_confidence, base_probabilities = self.predict_with_transformer(text, base_model, base_tokenizer)
                    base_model_name = f"Base {model_type}"
            
            # Update titles
            self.results_title_left.config(text=f"Finetuned Model Results")
            self.results_title_right.config(text=f"Base Model Results")
            
            # Display results
            self.display_sentiment_results(
                self.results_content_left, 
                finetuned_sentiment, 
                finetuned_confidence, 
                finetuned_probabilities, 
                star_rating=star_rating,
                model_name=finetuned_model_name
            )
            
            self.display_sentiment_results(
                self.results_content_right, 
                base_sentiment, 
                base_confidence, 
                base_probabilities, 
                star_rating=star_rating,
                model_name=base_model_name
            )
            
            # Display aspect analysis on left frame
            self.display_aspect_analysis(aspects, self.results_content_left)
            
            # Display highlighted review on right frame
            self.display_highlighted_review(text, aspects, self.results_content_right)
            
        except Exception as e:
            # Show error message
            self.results_title_left.config(text="Analysis Results")
            self.results_title_right.config(text="Analysis Results")
            
            error_label_left = tk.Label(self.results_content_left, text=f"Error: {str(e)}", 
                                     font=("Arial", 12), fg="red", bg="#F9F9F9", wraplength=200)
            error_label_left.pack(pady=20)
            
            error_label_right = tk.Label(self.results_content_right, text=f"Error: {str(e)}", 
                                      font=("Arial", 12), fg="red", bg="#F9F9F9", wraplength=200)
            error_label_right.pack(pady=20)
            
            self.results_content_left.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.results_content_right.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            traceback_info = traceback.format_exc()
            print(f"Detailed error:\n{traceback_info}")

# Add necessary imports
import torch
import torch.nn.functional as F
import traceback

if __name__ == "__main__":
    # Set up nicer theme for ttk widgets
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()
        print("ttkthemes not installed. Using default theme.")
    
    # Create app
    app = AmazonSentimentApp(root)
    
    # Start main loop
    root.mainloop()