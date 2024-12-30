from restack_ai.workflow import workflow, log, workflow_info, import_functions
from datetime import timedelta
import json

from .child_workflow_a import ChildWorkflowA
from .child_workflow_b import ChildWorkflowB

with import_functions():
    from src.functions.decide import decide, DecideInput

@workflow.defn()
class ParentWorkflow:
    @workflow.run
    async def run(self):
        parent_workflow_id = workflow_info().workflow_id

        decide_result = await workflow.step(
            decide,
            input=DecideInput(
                email="john.doe@restack.io",
                current_accepted_applicants_count=9
            ),
            start_to_close_timeout=timedelta(seconds=120)
        )

        decision = json.loads(decide_result)["accepted"]

        child_workflow_result = None
        if decision:
            child_workflow_result = await workflow.child_execute(
                ChildWorkflowA,
                workflow_id=f"{parent_workflow_id}-child-a",
            )
        elif not decision:
            child_workflow_result = await workflow.child_execute(
                ChildWorkflowB,
                workflow_id=f"{parent_workflow_id}-child-b",
            )

        return child_workflow_result


