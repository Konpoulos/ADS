
##########################################################
######## Example 5: Different Data Visualization Styles ##
##########################################################


library(shinydashboard)
library(DT)




##  content
u <- dashboardPage(
  dashboardHeader(title = "My Dashboard"),
  ## Sidebar content
  dashboardSidebar(
    sidebarMenu(
      menuItem("Select Data", tabName = "load"),
      menuItem("Data Exploration", tabName = "explore"),
      menuItem("Outlier Detection", tabName = "od"),
      menuItem("Clustering", tabName = "cluster")
    )
  ),
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "load",
              sidebarPanel(
                #Selector for file upload
                fileInput('datafile', 'Choose CSV file',
                          accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv'))
              ),
              mainPanel(
                DT::dataTableOutput("filetable")
              )
      ),
      
      # Second tab content
      tabItem(tabName = "explore",
              selectInput("visStyle", "Visualization Style", c("Select an option .. ", "Scatter", "Histogram", "Tabular")),
              conditionalPanel(condition = "(input.visStyle == 'Scatter')",
                               sidebarPanel(
                                 uiOutput("expCol"),
                                 selectInput(inputId = 'shape', label = "Point Shape", seq(from = 1, to = 20), selected = 5),
                                 selectInput(inputId = 'color', label = "Point Color", seq(from = 1, to = 20), selected = 2),
                                 selectInput(inputId = 'msize', label = "Marker Size", seq(from = 1, to = 10), selected = 2),
                                 selectInput(inputId = 'lwidth', label = "Border Line Width", seq(from = 1, to = 10), selected = 2)
                               ),
                               mainPanel(
                                 plotOutput('scatterPlot')
                               ),
              ),
              conditionalPanel(condition = "(input.visStyle == 'Histogram')",
                               sidebarPanel(
                                 uiOutput("expCol2"),
                                 sliderInput(inputId = 'n_bins', label = 'Number of histogram bins', value = 20,
                                             min = 5, max = 25, step = 5)
                                 
                               ),
                               mainPanel(
                                 plotOutput('histPlot')
                               )
              )
      ),
      
      # Third tab content
      tabItem(tabName = "od",
              tabsetPanel(
                # The id lets us use input$tabset1 on the server to find the current tab
                id = "tabset1",
                tabPanel("Statistical Outlier Detection",  "TODO: implement statitical based outlier detection"),
                tabPanel("Distance-Based Outleir Detection", "TODO: implement distance-based outlier detection")
              ),
      ),
      
      # Fourth tab content
      tabItem(tabName = "cluster",
              tabsetPanel(
                # The id lets us use input$tabset1 on the server to find the current tab
                id = "tabset2",
                tabPanel("kmeans",  "TODO: implement kmeans clustering demo"),
                tabPanel("DBSCAN", "TODO: implement DBSCAN clustering demo")
              ),
      )
    )
  )
  
)

s <- function(input, output) {
  #This function is repsonsible for loading in the selected file
  filedata <- reactive({
    infile <- input$datafile
    if (is.null(infile)) {
      # User has not uploaded a file yet
      return(NULL)
    }
    read.csv(infile$datapath)
  })
  
  
  #The following set of functions populate the column selectors
  output$expCol <- renderUI({
    df <-filedata()
    if (is.null(df)) return(NULL)
    
    items=names(df)
    names(items)=items
    selectInput(inputId = "selectedCol", label = "Seclect Column:",items)
    
  })
  
  output$expCol2 <- renderUI({
    df <-filedata()
    if (is.null(df)) return(NULL)
    
    items=names(df)
    names(items)=items
    selectInput(inputId = "selectedCol2", label = "Seclect Column:",items)
    
  })
  
  #This previews the CSV data file
  output$filetable <- DT::renderDataTable({
    df <-filedata()
    if (is.null(df)) return(NULL)
    data.frame(names(df))
  }
  )
  
  
  # This renders the scatter plot for the selected column
  
  output$scatterPlot <- renderPlot({
    df <-filedata()
    if (is.null(df)) return(NULL)
    if (input$visStyle == 'Scatter'){
      selectedData <- df[, input$selectedCol]
      if(is.numeric(selectedData)){
        plot(selectedData, pch = as.numeric(input$shape), col = as.numeric(input$color),
             cex = as.numeric(input$msize), lwd = input$lwidth)
      }
      else{
        return(NULL)
      }
    }
    else{
      return(NULL)
    }
  })
  
  
  output$histPlot <- renderPlot({
    df <-filedata()
    if (is.null(df)) return(NULL)
    if (input$visStyle == 'Histogram'){
      selectedData <- df[, input$selectedCol2]
      if (is.numeric(selectedData)){
        hist(selectedData, breaks = as.numeric(input$n_bins))
      }
      else{
        return(NULL)
      }
    }
    else{
      return(NULL)
    }
  })
  
  
}

shinyApp(ui = u, server = s)

