pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // repositorio de nightlies de TensorFlow
        maven { url = uri("https://storage.googleapis.com/tensorflow/android-repo/nightly") }
    }
}

rootProject.name = "SmartRecycle"
include(":app")
 