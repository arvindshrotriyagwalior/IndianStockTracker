                with tab5:
                    st.subheader("News & Market Sentiment Analysis")
                    
                    with st.spinner("Fetching and analyzing news..."):
                        # Get news and sentiment
                        news_df, sentiment_metrics = get_news_sentiment(symbol, max_articles=5)
                        
                        if news_df.empty:
                            st.error("No news articles found for this stock. Try another symbol or check your internet connection.")
                        else:
                            # Display sentiment summary
                            st.markdown("### Market Sentiment Summary")
                            
                            # Determine sentiment color
                            sentiment_color = "gray"
                            if sentiment_metrics['overall_sentiment'] == 'positive':
                                sentiment_color = "green"
                            elif sentiment_metrics['overall_sentiment'] == 'negative':
                                sentiment_color = "red"
                            
                            # Create sentiment metrics - adjust for mobile
                            # Import is_mobile function from main app
                            from app import is_mobile
                            
                            # Use 2 columns for mobile, 3 for desktop
                            if is_mobile():
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style='background-color: rgba(240,240,240,0.8); padding: 10px; border-radius: 5px; text-align: center;'>
                                        <h4 style='color: {sentiment_color}; margin: 0; font-size: 0.9rem;'>{sentiment_metrics['overall_sentiment'].upper()}</h4>
                                        <p style='font-size: 0.8rem;'>Overall Sentiment</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Create sentiment score display
                                    score = sentiment_metrics['overall_score']
                                    score_percentage = (score + 1) / 2 * 100  # Convert -1 to 1 scale to 0-100%
                                    
                                    st.markdown(f"""
                                    <div style='background-color: rgba(240,240,240,0.8); padding: 10px; border-radius: 5px; text-align: center;'>
                                        <h4 style='color: {sentiment_color}; margin: 0; font-size: 0.9rem;'>{score:.2f}</h4>
                                        <p style='font-size: 0.8rem;'>Sentiment Score</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Articles analyzed in a full-width row for mobile
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: 10px; border-radius: 5px; text-align: center; margin-top: 10px;'>
                                    <h4 style='margin: 0; font-size: 0.9rem;'>{sentiment_metrics['total_articles']}</h4>
                                    <p style='font-size: 0.8rem;'>Articles Analyzed</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Desktop layout with 3 columns
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                        <h4 style='color: {sentiment_color}; margin: 0;'>{sentiment_metrics['overall_sentiment'].upper()}</h4>
                                        <p>Overall Sentiment</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Create sentiment score display
                                    score = sentiment_metrics['overall_score']
                                    score_percentage = (score + 1) / 2 * 100  # Convert -1 to 1 scale to 0-100%
                                    
                                    st.markdown(f"""
                                    <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                        <h4 style='color: {sentiment_color}; margin: 0;'>{score:.2f}</h4>
                                        <p>Sentiment Score</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                        <h4 style='margin: 0;'>{sentiment_metrics['total_articles']}</h4>
                                        <p>Articles Analyzed</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Create sentiment distribution chart
                            st.markdown("### Sentiment Distribution")
                            
                            sentiment_dist = {
                                'Positive': sentiment_metrics['positive_articles'],
                                'Neutral': sentiment_metrics['neutral_articles'],
                                'Negative': sentiment_metrics['negative_articles']
                            }
                            
                            # Create chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=list(sentiment_dist.keys()),
                                y=list(sentiment_dist.values()),
                                marker_color=['rgba(0, 200, 0, 0.6)', 'rgba(180, 180, 180, 0.6)', 'rgba(200, 0, 0, 0.6)']
                            ))
                            
                            # Adjust chart height for mobile
                            chart_height = 250 if is_mobile() else 300
                            margin_settings = dict(t=30, b=30, l=30, r=20) if is_mobile() else dict(t=50, b=50, l=50, r=50)
                            
                            fig.update_layout(
                                title="News Sentiment Distribution",
                                xaxis_title="Sentiment",
                                yaxis_title="Number of Articles",
                                height=chart_height,
                                plot_bgcolor='rgba(240,240,240,0.8)',
                                paper_bgcolor='rgba(255,255,255,1)',
                                font=dict(
                                    family="Arial, sans-serif", 
                                    size=10 if is_mobile() else 12, 
                                    color="#505050"
                                ),
                                margin=margin_settings
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display recent news with sentiment
                            st.markdown("### Recent News Articles")
                            
                            # Format and display each news article
                            for i, news in news_df.iterrows():
                                sentiment_icon = "🔍"
                                if news['sentiment'] == 'positive':
                                    sentiment_icon = "📈"
                                elif news['sentiment'] == 'negative':
                                    sentiment_icon = "📉"
                                elif news['sentiment'] == 'neutral':
                                    sentiment_icon = "⚖️"
                                
                                # Format date
                                date_str = news['date'].strftime('%b %d, %Y')
                                
                                # Adjust the news card style based on device
                                padding = "10px" if is_mobile() else "15px"
                                font_size = "0.9rem" if is_mobile() else "1rem"
                                title_font_size = "1rem" if is_mobile() else "1.2rem"
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: {padding}; border-radius: 5px; margin-bottom: 10px;'>
                                    <h4 style='margin-top: 0; font-size: {title_font_size};'>{sentiment_icon} {news['title']}</h4>
                                    <p style='font-size: {font_size};'><strong>Source:</strong> {news['source']} | <strong>Date:</strong> {date_str}</p>
                                    <p style='font-size: {font_size};'><strong>Sentiment:</strong> {news['sentiment'].capitalize()} (Score: {news['score']:.2f})</p>
                                    <a href="{news['url']}" target="_blank" style='font-size: {font_size};'>Read full article</a>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add sentiment impact on stock
                            st.markdown("### Sentiment Impact")
                            
                            # Provide interpretation based on sentiment
                            if sentiment_metrics['overall_sentiment'] == 'positive':
                                st.success("""
                                **Positive Market Sentiment**: The recent news coverage for this stock is predominantly positive, 
                                which may indicate favorable market perception. Positive news can potentially drive buying interest and price appreciation.
                                """)
                            elif sentiment_metrics['overall_sentiment'] == 'negative':
                                st.error("""
                                **Negative Market Sentiment**: The recent news coverage for this stock shows concerning signals, 
                                with negative sentiment dominating. This could lead to selling pressure in the near term.
                                """)
                            else:
                                st.info("""
                                **Neutral Market Sentiment**: The recent news coverage for this stock is balanced or neutral, 
                                suggesting no strong directional bias from news sentiment alone. Other factors may have more influence on price movements.
                                """)
                            
                            # Add cross-analysis with technical indicators
                            if signals and 'rec' in locals():
                                st.markdown("### Sentiment vs. Technical Analysis")
                                
                                # Compare sentiment with technical indicators
                                if (sentiment_metrics['overall_sentiment'] == 'positive' and rec == 'BUY') or \
                                   (sentiment_metrics['overall_sentiment'] == 'negative' and rec == 'SELL'):
                                    st.success("""
                                    **Strong Confirmation**: Market sentiment aligns with technical analysis signals, 
                                    providing stronger confirmation for the trading recommendation.
                                    """)
                                elif (sentiment_metrics['overall_sentiment'] == 'positive' and rec == 'SELL') or \
                                     (sentiment_metrics['overall_sentiment'] == 'negative' and rec == 'BUY'):
                                    st.warning("""
                                    **Mixed Signals**: Market sentiment contradicts technical analysis signals. 
                                    This divergence suggests caution and more thorough research before taking action.
                                    """)
                                else:
                                    st.info("""
                                    **Neutral Alignment**: Market sentiment is neutral relative to technical signals. 
                                    Consider prioritizing technical factors in your decision-making.
                                    """)
                            
                            # Disclaimer about news analysis
                            st.warning("""
                            **DISCLAIMER:** News sentiment analysis is based on a basic keyword-based sentiment algorithm, 
                            not comprehensive natural language processing. Results should be considered approximations and not definitive market sentiment indicators.
                            Always perform thorough research and consider multiple factors before making investment decisions.
                            """)