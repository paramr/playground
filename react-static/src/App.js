import React, { PureComponent } from 'react'
import { Root, Routes } from 'react-static'
import { Link } from '@reach/router'

import CssBaseline from '@material-ui/core/CssBaseline'
import AppBar from '@material-ui/core/AppBar'
import Tabs from '@material-ui/core/Tabs'
import Tab from '@material-ui/core/Tab'
import { withStyles } from '@material-ui/core/styles'

// Custom styles
const styles = {
  '@global': {
    img: {
      maxWidth: '100%',
    },
  },
  appBar: {
    flexWrap: 'wrap',
  },
  tabs: {
    width: '100%',
  },
  content: {
    padding: '1rem',
  },
}

class App extends PureComponent {
  // Remove the server-side injected CSS.
  componentDidMount() {
    const jssStyles = document.getElementById('jss-server-side')
    if (jssStyles && jssStyles.parentNode) {
      jssStyles.parentNode.removeChild(jssStyles)
    }
  }


  render() {
    const { classes } = this.props

/*
    return (
    <Root>
      <AppBar className={classes.appBar} color="default" position="static">
      <nav>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
      </nav>
      </AppBar>
      <div className="content">
        <Routes />
      </div>
    </Root>
    )*/
    return (
      <Root>
        <div className={classes.container}>
          <CssBaseline />
          <AppBar className={classes.appBar} color="default" position="static">
            <nav>
              <Tabs className={classes.tabs} value={false}>
                <Tab component={Link} to="/" label="Home" />
                <Tab component={Link} to="/about" label="About" />
              </Tabs>
            </nav>
          </AppBar>
          <div className={classes.content}>
            <Routes />
          </div>
        </div>
      </Root>
    )
/*
    return (
      <Root>
        <div className={classes.container}>
          <CssBaseline />
          <AppBar className={classes.appBar} color="default" position="static">
            <nav>
              <Tabs>
              </Tabs>
            </nav>
          </AppBar>
          <div>
            <Routes />
          </div>
        </div>
      </Root>
    )*/
  }
}

const AppWithStyles = withStyles(styles)(App)

export default AppWithStyles
// export default App
