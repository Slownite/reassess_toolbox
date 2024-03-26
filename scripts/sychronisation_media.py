

class SynchroniseurEEGVideo:
    def __init__(self, video_path, egg_path, frequence_echantillonnage_eeg):
        self.video_path = video_path
        self.chemin_eeg = egg_path
        self.frequence_echantillonnage_eeg = frequence_echantillonnage_eeg
        # Structures de données pour stocker le contenu chargé
        self.frames_video = []
        self.donnees_eeg = []