# Dictionary mapping cluster IDs to their base marketing strategy prompts.
marketing_prompts = {
    0: (
        "Cluster 0 represents high spenders. Our strategy for this cluster focuses on premium offers, "
        "exclusive loyalty programs, and personalized high-value incentives. "
        "Provide a detailed recommendation, including quantitative methods to track engagement, conversion, and retention."
    ),
    1: (
        "Cluster 1 represents moderate spenders. The approach is to use personalized discount codes and targeted messaging. "
        "Recommend actionable strategies and key performance metrics for optimizing campaign performance."
    ),
    2: (
        "Cluster 2 represents low spenders. Our goal is to stimulate increased usage with introductory offers and re-engagement campaigns. "
        "Please provide a recommendation that includes suggestions for measuring improvements and boosting customer activity."
    ),
    3: (
        "Cluster 3 represents frequent cash advance users. This segment requires proactive financial advice and repayment incentives. "
        "Outline a strategy with quantitative methods to monitor risk and encourage better financial behavior."
    ),
    4: (
        "Cluster 4 represents occasional users. Focus on re-engagement campaigns to drive more consistent usage and improve customer lifetime value. "
        "Provide a detailed marketing recommendation with appropriate tracking metrics."
    )
}

def get_prompt_for_cluster(cluster_id: int, additional_context: str = "") -> str:
    """
    Retrieves the marketing prompt for a specific cluster.
    
    Parameters:
        cluster_id (int): The ID of the customer cluster.
        additional_context (str): Any additional context to append to the base prompt.
        
    Returns:
        str: The complete marketing prompt.
    """
    base_prompt = marketing_prompts.get(cluster_id, "No strategy available for this cluster.")
    if additional_context:
        return f"{base_prompt}\nAdditional Context: {additional_context}"
    return base_prompt