# DependencyInjection/container.py
from dependency_injector import containers, providers
from DataExtractors.WordExtracter import WordExtractor
from DataExtractors.PDFExtractor import PdfExtractor
from HelperMethods.PDFValidator import PDFValidator
from HelperMethods.WordValidator import WordValidator
from Models.MlModels import MlModels


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["__main__"])

    # Validators
    pdf_validator = providers.Singleton(PDFValidator)
    word_validator = providers.Singleton(WordValidator)

    # Extractors
    pdf_extractor = providers.Singleton(
        PdfExtractor,
        validator=pdf_validator
    )

    word_extractor = providers.Singleton(
        WordExtractor,
        validator=word_validator
    )

    #ML Models
    ml_models = providers.Singleton(MlModels)