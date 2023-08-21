import qtawesome as qta

from nn_visualizer.application_context.application_context import (
    ApplicationContext,
    ApplicationState,
)
from nn_visualizer.ui_components.customized.button.smlf_icon_button import SMLFIconButton
from nn_visualizer.ui_components.customized.button.smlf_icon_round_button import SMLFIconRoundButton


class PlayFullButton(SMLFIconRoundButton):
    def __init__(self, application_context: ApplicationContext):
        SMLFIconRoundButton.__init__(
            self,
            icon=qta.icon('mdi.play', color='white'),
            click_handler=self.handler
        )
        self.application_context = application_context

    def handler(self, event):
        icon = None
        if self.application_context.state == ApplicationState.PLAYING_FULL_TRAINING:
            icon = qta.icon('mdi.play', color='white')
            self.setIcon(icon)
            self.update()
            # self.application_context.play_full_pipeline(event)
        if self.application_context.state == ApplicationState.IDLE_TRAINING:
            icon = qta.icon('mdi.pause', color='white')
            # self.application_context.stop_full_pipeline(event)
            self.setIcon(icon)
            self.update()

        self.application_context.play_training_handler(event)
