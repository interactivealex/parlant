# healthcare_postgres.py
# Runs the healthcare example against PostgreSQL + pgvector

import parlant.sdk as p
import asyncio

from healthcare import (
    add_domain_glossary,
    create_scheduling_journey,
    create_lab_results_journey,
    get_insurance_providers,
)

PG_DSN = "postgresql://parlant:parlant@localhost:5432/parlant"


async def main() -> None:
    async with p.Server(
        session_store=PG_DSN,
        customer_store=PG_DSN,
        variable_store=PG_DSN,
        log_level=p.LogLevel.DEBUG,
    ) as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )

        await add_domain_glossary(agent)
        scheduling_journey = await create_scheduling_journey(server, agent)
        lab_results_journey = await create_lab_results_journey(server, agent)

        status_inquiry = await agent.create_observation(
            "The patient asks to follow up on their visit, but it's not clear in which way",
        )

        await status_inquiry.disambiguate([scheduling_journey, lab_results_journey])

        await agent.create_guideline(
            condition="The patient asks about insurance",
            action="List the insurance providers we accept, and tell them to call the office for more details",
            tools=[get_insurance_providers],
        )

        await agent.create_guideline(
            condition="The patient asks to talk to a human agent",
            action="Ask them to call the office, providing the phone number",
        )

        await agent.create_guideline(
            condition="The patient inquires about something that has nothing to do with our healthcare",
            action="Kindly tell them you cannot assist with off-topic inquiries - do not engage with their request.",
        )


if __name__ == "__main__":
    asyncio.run(main())
