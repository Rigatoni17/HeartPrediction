library(shiny)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(ggplot2)
library(klaR) # Ensure klaR is available for method="nb" in caret

ui <- fluidPage(
  titlePanel("Heart Disease Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      radioButtons("dataSource", "Data Source:",
                   choices = c("Use preset file" = "preset",
                               "Upload your own file" = "upload"),
                   selected = "preset"),
      
      conditionalPanel(
        condition = "input.dataSource == 'upload'",
        fileInput("datafile", "Upload your CSV file:", accept = c(".csv"))
      ),
      
      sliderInput("splitRatio", "Training Split Ratio:", 
                  min = 0.5, max = 0.9, value = 0.7, step = 0.05),
      
      selectInput("modelType", "Select Model:", choices = c("rpart", "naive_bayes")),
      
      actionButton("trainButton", "Train Model")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Summary", verbatimTextOutput("dataSummary")),
        tabPanel("Model Summary", verbatimTextOutput("modelSummary"), plotOutput("modelPlot")),
        tabPanel("Confusion Matrix (Train)", verbatimTextOutput("cmTrain")),
        tabPanel("Confusion Matrix (Test)", verbatimTextOutput("cmTest")),
        tabPanel("Cross Validation Results", verbatimTextOutput("cvResults")),
        tabPanel("Data Exploration",
                 plotOutput("sexPlot"),
                 plotOutput("cpPlot"),
                 plotOutput("cholPlot"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  data_reactive <- reactive({
    if (input$dataSource == "preset") {
      validate(
        need(file.exists("heart.csv"), "The file heart.csv is not found in the current directory.")
      )
      df <- read.csv("heart.csv")
    } else {
      req(input$datafile)
      df <- read.csv(input$datafile$datapath)
    }
    # Ensure factor variables (adjust as needed based on your dataset's columns)
    df$HeartDisease <- as.factor(df$HeartDisease)
    if("Sex" %in% names(df)) df$Sex <- as.factor(df$Sex)
    if("ChestPainType" %in% names(df)) df$ChestPainType <- as.factor(df$ChestPainType)
    if("FastingBS" %in% names(df)) df$FastingBS <- as.factor(df$FastingBS)
    
    df
  })
  
  rv <- reactiveValues(
    rpartModel = NULL,
    nbModel = NULL,
    training_set = NULL,
    test_set = NULL,
    predTrain = NULL,
    predTest = NULL,
    cvResults = NULL,
    errorMsg = NULL
  )
  
  observeEvent(input$trainButton, {
    rv$errorMsg <- NULL
    df <- data_reactive()
    set.seed(123)
    
    training_size <- floor(input$splitRatio * nrow(df))
    training_indices <- sample(seq_len(nrow(df)), size = training_size)
    training_set <- df[training_indices,]
    test_set <- df[-training_indices,]
    
    # Ensure test set has same factor levels as training set for categorical vars:
    for(col_name in names(training_set)) {
      if(is.factor(training_set[[col_name]]) && col_name %in% names(test_set)) {
        test_set[[col_name]] <- factor(test_set[[col_name]], levels = levels(training_set[[col_name]]))
      }
    }
    
    # Train rpart model with model=TRUE
    rpmodel <- rpart(HeartDisease ~ ., data = training_set, method = "class",
                     model = TRUE,
                     control = rpart.control(cp = 0.006920415))
    
    # Attempt to train Naive Bayes model with tryCatch to capture errors:
    nbmodel <- tryCatch({
      suppressWarnings({
        naiveBayes(HeartDisease ~ ., data = training_set)
      })
    }, error = function(e) {
      rv$errorMsg <- paste("Naive Bayes model training error:", e$message)
      NULL
    })
    
    rv$rpartModel <- rpmodel
    rv$nbModel <- nbmodel
    rv$training_set <- training_set
    rv$test_set <- test_set
    
    # Predictions based on model selection (with tryCatch for NB predictions)
    if (input$modelType == "rpart") {
      rv$predTrain <- predict(rpmodel, training_set, type = "class")
      rv$predTest <- predict(rpmodel, test_set, type = "class")
    } else {
      # Only attempt predictions if nbModel is not NULL
      if (!is.null(nbmodel)) {
        rv$predTrain <- tryCatch({
          suppressWarnings(predict(nbmodel, training_set, type = "class"))
        }, error = function(e) {
          rv$errorMsg <- paste("Naive Bayes train prediction error:", e$message)
          NULL
        })
        
        rv$predTest <- tryCatch({
          suppressWarnings(predict(nbmodel, test_set, type = "class"))
        }, error = function(e) {
          rv$errorMsg <- paste("Naive Bayes test prediction error:", e$message)
          NULL
        })
      }
    }
    
    # If no errors so far, try cross-validation
    if (is.null(rv$errorMsg)) {
      train_control <- trainControl(method = "cv", number = 5, savePredictions = TRUE)
      rv$cvResults <- tryCatch({
        if (input$modelType == "rpart") {
          train(HeartDisease ~ ., data = training_set, trControl=train_control, method="rpart")
        } else {
          suppressWarnings({
            train(HeartDisease ~ ., data = training_set, trControl=train_control, method="nb")
          })
        }
      }, error = function(e) {
        rv$errorMsg <- paste("Cross-validation error:", e$message)
        NULL
      })
    } else {
      rv$cvResults <- NULL
    }
  })
  
  output$dataSummary <- renderPrint({
    req(rv$training_set)
    summary(data_reactive())
  })
  
  output$modelSummary <- renderPrint({
    # If there's an error message, print it:
    if (!is.null(rv$errorMsg)) {
      cat(rv$errorMsg, "\n")
      return()
    }
    req(rv$rpartModel, rv$nbModel)
    if (input$modelType == "rpart") {
      print(rv$rpartModel)
    } else {
      if (!is.null(rv$nbModel)) {
        summary(rv$nbModel)
      } else {
        cat("No Naive Bayes model trained due to an error.\n")
      }
    }
  })
  
  output$modelPlot <- renderPlot({
    req(rv$rpartModel)
    if (input$modelType == "rpart") {
      rpart.plot(rv$rpartModel)
    } else {
      if (!is.null(rv$nbModel)) {
        preds <- suppressWarnings({
          predict(rv$nbModel, rv$training_set, type="raw")
        })
        hist(preds[,2], main="Naive Bayes Predicted Probabilities", xlab="P(HeartDisease=1)")
      }
    }
  })
  
  output$cmTrain <- renderPrint({
    # Show confusion matrix only if no errors and predictions exist
    if (!is.null(rv$errorMsg)) {
      cat(rv$errorMsg, "\n")
      return()
    }
    req(rv$predTrain, rv$training_set)
    confusionMatrix(rv$predTrain, rv$training_set$HeartDisease)
  })
  
  output$cmTest <- renderPrint({
    if (!is.null(rv$errorMsg)) {
      cat(rv$errorMsg, "\n")
      return()
    }
    req(rv$predTest, rv$test_set)
    confusionMatrix(rv$predTest, rv$test_set$HeartDisease)
  })
  
  output$cvResults <- renderPrint({
    if (!is.null(rv$errorMsg)) {
      cat(rv$errorMsg, "\n")
      return()
    }
    req(rv$cvResults)
    rv$cvResults
  })
  
  # Data Exploration Plots
  output$sexPlot <- renderPlot({
    df <- data_reactive()
    ggplot(df, aes(x = Sex, fill = HeartDisease)) +
      geom_bar(position = "dodge") +
      labs(title = "Heart Disease by Sex", x = "Sex", y = "Count", fill = "HeartDisease") +
      theme_minimal()
  })
  
  output$cpPlot <- renderPlot({
    df <- data_reactive()
    ggplot(df, aes(x = ChestPainType, fill = HeartDisease)) +
      geom_bar(position = "dodge") +
      labs(title = "Heart Disease by Chest Pain Type", x = "Chest Pain Type", y = "Count", fill = "HeartDisease") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$cholPlot <- renderPlot({
    df <- data_reactive()
    ggplot(df, aes(x = HeartDisease, y = Cholesterol, fill = HeartDisease)) +
      geom_boxplot() +
      labs(title = "Cholesterol Distribution by Heart Disease Status", 
           x = "Heart Disease", y = "Cholesterol") +
      theme_minimal() +
      theme(legend.position = "none")
  })
}

shinyApp(ui = ui, server = server)
