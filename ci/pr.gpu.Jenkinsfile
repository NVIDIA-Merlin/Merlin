pipeline {
    agent {
        docker {
            image 'nvcr.io/nvstaging/merlin/merlin-ci-runner-wrapper'
            label 'merlin_gpu'
            registryCredentialsId 'jawe-nvcr-io'
            registryUrl 'https://nvcr.io'
            args "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size '256m'"
        }
    }

    options {
      buildDiscarder(logRotator(numToKeepStr: '10'))
      ansiColor('xterm')
      disableConcurrentBuilds(abortPrevious: true)
    }

    stages {
        stage("dummy stage") {
            options {
                timeout(time: 60, unit: 'MINUTES', activity: true)
            }
            steps {
                sh """#!/bin/bash
                echo 'dummy stage'
                """
            }
        }
    }
}