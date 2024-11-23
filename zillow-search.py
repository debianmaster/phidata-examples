from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

web_agent = Agent(
    name="Real Estate Buyer",
    role="Search the web for given properties",
    instructions=["""
                Given the following properties , sort by lot size , year built and distance closer to cleveland city center. and print results in tabular format.
                  """],
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGo(fixed_max_results=15)],
    markdown=True,
    show_tool_calls=True,
)

web_agent.print_response("""
                1. https://www.zillow.com/homedetails/2675-River-Rd-Willoughby-OH-44094/34521191_zpid/
                2. https://www.zillow.com/homedetails/2106-Glenridge-Rd-Euclid-OH-44117/33637949_zpid/
                3. https://www.zillow.com/homedetails/4527-Monticello-Blvd-Cleveland-OH-44143/33664751_zpid/
                4. https://www.zillow.com/homedetails/164-Richmond-Rd-Richmond-Heights-OH-44143/33640324_zpid/
                5. https://www.zillow.com/homedetails/23400-Emery-Rd-Warrensville-Heights-OH-44128/33698955_zpid/
                6. https://www.zillow.com/homedetails/4258-Bluestone-Rd-South-Euclid-OH-44121/33665586_zpid/
                7. https://www.zillow.com/homedetails/2042-Taylor-Rd-East-Cleveland-OH-44112/33648453_zpid/
                8. https://www.zillow.com/homedetails/19204-Shawnee-Ave-Cleveland-OH-44119/71969405_zpid/
                9. https://www.zillow.com/homedetails/448-Harris-Rd-Richmond-Heights-OH-44143/33641746_zpid/
                10. https://www.zillow.com/homedetails/600-Lloyd-Rd-Euclid-OH-44132/33634119_zpid/
                         """, stream=True)

